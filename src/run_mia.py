import attacks
import datasetup
import hypertuner
import evaluation
import trainer
import utils

import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from statistics import mean, stdev
from collections import defaultdict

class MembershipInferenceExperiment:

    def __init__(self, config):
        self.config = utils.Config(config)
        self.dataset = datasetup.parse_dataset(root=self.config.datadir, name=self.config.dataset)
        self.criterion = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(self.config.device)
        print(utils.GraphInfo(self.dataset))

    def visualize_embedding_distribution(self):
        config = self.config
        dataset = datasetup.sample_subgraph(self.dataset, self.dataset.x.shape[0])
        savepath = f'{config.savedir}/embeddings/features.png'
        train_mask = ~dataset.test_mask
        utils.plot_embedding_2D_scatter(dataset.x, dataset.y, train_mask, savepath=savepath)
        target_model = self.train_target_model(dataset)
        target_model.eval()
        query_nodes = torch.arange(0, dataset.x.shape[0])
        savedir = f'{config.savedir}/embeddings'
        query_hops = config.query_hops
        savepath = f'{savedir}/emb_scatter_2D_{query_hops}hops_{config.name}.png'
        with torch.inference_mode():
            embs = evaluation.k_hop_query(
                model=target_model,
                dataset=dataset,
                query_nodes=query_nodes,
                num_hops=query_hops,
            )
            hinge = utils.hinge_loss(embs, dataset.y)
            utils.plot_embedding_2D_scatter(embs=embs, y=dataset.y, train_mask=train_mask, savepath=savepath)
            for label in range(dataset.num_classes):
                label_mask = dataset.y == label
                savepath = f'{savedir}/hinge_hist_class{label}_{config.name}.png'
                utils.plot_hinge_histogram(hinge, label_mask=label_mask, train_mask=train_mask, savepath=savepath)

    def train_target_model(self, dataset, plot_training_results=True):
        config = self.config

        if self.config.grid_search:
            opt_hyperparams = hypertuner.grid_search(
                dataset=dataset,
                model_type=self.config.model,
                epochs=self.config.epochs_target,
                early_stopping=self.config.early_stopping,
                optimizer=self.config.optimizer,
                hidden_dim=self.config.hidden_dim_target,
                transductive=config.transductive,
            )
            print(f'Grid search results: {opt_hyperparams}')
            lr, weight_decay, dropout = opt_hyperparams.values()
        else:
            lr, weight_decay, dropout = config.lr, config.weight_decay, config.dropout

        target_model = utils.fresh_model(
            model_type=self.config.model,
            num_features=dataset.num_features,
            hidden_dims=self.config.hidden_dim_target,
            num_classes=dataset.num_classes,
            dropout=dropout,
        )

        train_config = trainer.TrainConfig(
            criterion=self.criterion,
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=F.cross_entropy,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        train_res = trainer.train_gnn(
            model=target_model,
            dataset=dataset,
            config=train_config,
            inductive_split=not config.transductive
        )
        evaluation.evaluate_graph_training(
            model=target_model,
            dataset=dataset,
            criterion=train_config.criterion,
            training_results=train_res if plot_training_results else None,
            plot_title="Target model",
            savedir=config.savedir,
        )
        return target_model

    def run(self):
        config = self.config
        dataset = self.dataset
        scores = defaultdict(list)
        best_auroc = 0
        for i in range(config.experiments):
            print(f'Running experiment {i + 1}/{config.experiments}.')

            if config.attack == "basic-mlp":
                target_dataset, shadow_dataset = datasetup.target_shadow_split(dataset, split=config.split)
                target_model = self.train_target_model(target_dataset)
                metrics = attacks.BasicMLPAttack(
                    target_model=target_model,
                    shadow_dataset=shadow_dataset,
                    config=config,
                ).run_attack(target_samples=target_dataset)

            elif config.attack == "confidence":
                target_dataset = datasetup.sample_subgraph(dataset, num_nodes=dataset.x.shape[0]//2)
                target_model = self.train_target_model(target_dataset)
                metrics = attacks.ConfidenceAttack(
                    target_model=target_model,
                    config=config,
                ).run_attack(target_samples=target_dataset)

            elif config.attack == "lira":
                # In offline LiRA, the shadow models are trained on datasets that does not contain the target sample.
                # Therefore we make a disjoint split and train shadow models on one part, and attack samples of the other part.
                target_dataset, population = datasetup.target_shadow_split(dataset, split="disjoint", target_frac=0.5, shadow_frac=0.5)
                target_model = self.train_target_model(target_dataset)
                metrics = attacks.LiRA(
                    target_model=target_model,
                    population=population,
                    config=config,
                ).run_attack(target_samples=target_dataset)
                
            elif config.attack == "rmia":
                target_dataset, population = datasetup.target_shadow_split(dataset, split="disjoint", target_frac=0.5, shadow_frac=0.5)
                target_model = self.train_target_model(target_dataset)
                metrics = attacks.RMIA(
                    target_model=target_model,
                    population=population,
                    config=config,
                ).run_attack(target_samples=target_dataset)

            else:
                raise AttributeError(f"No attack named {config.attack}")

            target_scores = {
                'train_score': evaluation.evaluate_graph_model(
                    model=target_model,
                    dataset=target_dataset,
                    mask=target_dataset.train_mask,
                    criterion=self.criterion,
                ),
                'test_score': evaluation.evaluate_graph_model(
                    model=target_model,
                    dataset=target_dataset,
                    mask=target_dataset.test_mask,
                    criterion=self.criterion,
                ),
            }
            metrics = dict(target_scores, **metrics)

            fpr, tpr = metrics['roc']
            tpr_at_fixed_fpr = utils.tpr_at_fixed_fpr(fpr, tpr, config.target_fpr)

            scores['fprs'].append(fpr)
            scores['tprs'].append(tpr)
            scores['train_scores'].append(metrics['train_score'])
            scores['test_scores'].append(metrics['test_score'])
            scores['auroc'].append(metrics['auroc'])
            scores['tprs_at_fixed_fpr'].append(tpr_at_fixed_fpr)
            if best_auroc < metrics['auroc']:
                best_auroc = metrics['auroc']

        if config.experiments > 1:
            stats = {
                'train_acc_mean': [mean(scores['train_scores'])],
                'train_acc_stdev': [stdev(scores['train_scores'])],
                'test_acc_mean': [mean(scores['test_scores'])],
                'test_acc_stdev': [stdev(scores['test_scores'])],
                'auroc_mean': [mean(scores['auroc'])],
                'auroc_stdev': [stdev(scores['auroc'])],
                'tpr_at_fixed_fpr_mean': [mean(scores['tprs_at_fixed_fpr'])],
                'tpr_at_fixed_fpr_stdev': [stdev(scores['tprs_at_fixed_fpr'])]
            }
        else:
            stats = {
                'train_acc': scores['train_scores'],
                'test_acc': scores['test_scores'],
                'auroc': scores['auroc'],
                'tpr_at_fixed_fpr': scores['tprs_at_fixed_fpr'],
            }

        stat_df = pd.DataFrame(stats, index=[config.name])
        roc_df = pd.DataFrame({f'{config.name}_fpr': fpr, f'{config.name}_tpr': tpr}) # TODO: save fprs and tprs.
        if config.make_plots:
            savepath = f'{config.savedir}/{config.name}_roc_loglog.png'
            utils.plot_multi_roc_loglog(scores['fprs'], scores['tprs'], scores['train_scores'], scores['test_scores'], savepath=savepath)
        return stat_df, roc_df


def main(config):
    config['dataset'] = config['dataset'].lower()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    mie = MembershipInferenceExperiment(config)
    mie.visualize_embedding_distribution()
    return mie.run()

if __name__ == '__main__':
    torch.random.manual_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="confidence", type=str)
    parser.add_argument("--dataset", default="cora", type=str)
    parser.add_argument("--split", default="sampled", type=str)
    parser.add_argument("--transductive", action=argparse.BooleanOptionalAction)
    parser.add_argument("--model", default="GCN", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs-target", default=500, type=int)
    parser.add_argument("--epochs-attack", default=100, type=int)
    parser.add_argument("--grid-search", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction)
    parser.add_argument("--hidden-dim-target", default=[32], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--hidden-dim-attack", default=[256, 64], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--query-hops", default=0, type=int)
    parser.add_argument("--experiments", default=1, type=int)
    parser.add_argument("--target-fpr", default=0.01, type=float)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--num-shadow-models", default=128, type=int)
    parser.add_argument("--rmia-offline-interp-param", default=0.1, type=float)
    parser.add_argument("--name", default="unnamed", type=str)
    parser.add_argument("--datadir", default="./data", type=str)
    parser.add_argument("--savedir", default="./results", type=str)
    args = parser.parse_args()
    config = vars(args)
    config['make_plots'] = True
    print('Running MIA experiment.')
    print(utils.Config(config))
    print()
    stat_df, roc_df = main(config)
    print('Attack statistics:')
    print(stat_df)
    roc_df.to_csv(f'{args.savedir}/roc_{args.name}.csv', index=False)
