import attacks
import datasetup
import evaluation
import trainer
import utils

import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.datasets
from torchmetrics import Accuracy
from statistics import mean, stdev

class Config:

    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
    
    def __str__(self):
        return '\n'.join(f'{k}: {v}'.replace('_', ' ') for k, v in self.__dict__.items())

class MembershipInferenceExperiment:

    def __init__(self, config):
        self.config = Config(config)
        self.dataset = self.parse_dataset()
        self.target_model = utils.fresh_model(
            model_type=self.config.model,
            num_features=self.dataset.num_features,
            hidden_dim=self.config.hidden_dim_target,
            num_classes=self.dataset.num_classes,
            dropout=self.config.dropout,
        )
        self.criterion = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(self.config.device)

    def parse_dataset(self):
        config = self.config
        root = config.datadir
        if config.dataset == "cora":
            dataset = torch_geometric.datasets.Planetoid(root=root, name="Cora")
        elif config.dataset == "corafull":
            dataset = torch_geometric.datasets.CoraFull(root=root)
            dataset.name == "CoraFull"
        elif config.dataset == "citeseer":
            dataset = torch_geometric.datasets.Planetoid(root=root, name="CiteSeer")
        elif config.dataset == "chameleon":
            dataset = torch_geometric.datasets.WikipediaNetwork(root=root, name="chameleon")
        elif config.dataset == "pubmed":
            dataset = torch_geometric.datasets.Planetoid(root=root, name="PubMed")
        elif self.config.dataset == "flickr":
            dataset = torch_geometric.datasets.Flickr(root=root)
            dataset.name = "Flickr"
        else:
            raise ValueError("Unsupported dataset!")
        print(utils.GraphInfo(dataset))
        return dataset

    def train_target_model(self, dataset, plot_training_results=True):
        config = self.config
        
        train_config = trainer.TrainConfig(
            criterion=self.criterion,
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=F.cross_entropy,
            lr=config.lr,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        train_res = trainer.train_gnn(
            model=self.target_model,
            dataset=dataset,
            config=train_config,
        )
        evaluation.evaluate_graph_training(
            model=self.target_model,
            dataset=dataset,
            criterion=train_config.criterion,
            training_results=train_res if plot_training_results else None,
            plot_title="Target model",
            savedir=config.savedir,
        )

    def run(self):
        config = self.config
        dataset = self.dataset
        train_scores, test_scores = [], []
        aurocs = []
        best_auroc = 0
        best_roc = None
        fprs, tprs = [], []
        for i in range(config.experiments):
            print(f'Running experiment {i + 1}/{config.experiments}.')
            self.target_model.reset_parameters() # Reset and re-train target model for each experiment.

            if config.attack == "basic-shadow":
                target_dataset, shadow_dataset = datasetup.target_shadow_split(dataset, split=config.split)
                self.train_target_model(target_dataset)
                metrics = attacks.BasicShadowAttack(
                    target_model=self.target_model,
                    shadow_dataset=shadow_dataset,
                    config=config,
                ).run_attack(target_samples=target_dataset)

            elif config.attack == "confidence":
                target_dataset = datasetup.sample_subgraph(dataset, num_nodes=dataset.x.shape[0]//2)
                self.train_target_model(target_dataset)
                metrics = attacks.ConfidenceAttack(
                    target_model=self.target_model,
                    config=config,
                ).run_attack(target_samples=target_dataset)

            elif config.attack == "lira":
                # In offline LiRA, the shadow models are trained on datasets that does not contain the target sample.
                # Therefore we make a disjoint split and train shadow models on one part, and attack samples of the other part.
                target_dataset, population = datasetup.target_shadow_split(dataset, split="disjoint", target_frac=0.5, shadow_frac=0.5)
                self.train_target_model(target_dataset)
                metrics = attacks.LiRA(
                    target_model=self.target_model,
                    population=population,
                    config=config,
                ).run_attack(target_samples=target_dataset)
                
            elif config.attack == "rmia":
                target_dataset, population = datasetup.target_shadow_split(dataset, split="disjoint", target_frac=0.5, shadow_frac=0.5)
                self.train_target_model(target_dataset)
                metrics = attacks.RMIA(
                    target_model=self.target_model,
                    population=population,
                    config=config,
                ).run_attack(target_samples=target_dataset)

            else:
                raise AttributeError(f"No attack named {config.attack}")

            target_scores = {
                'train_score': evaluation.evaluate_graph_model(
                    model=self.target_model,
                    dataset=target_dataset,
                    mask=target_dataset.train_mask,
                    criterion=self.criterion,
                ),
                'test_score': evaluation.evaluate_graph_model(
                    model=self.target_model,
                    dataset=target_dataset,
                    mask=target_dataset.test_mask,
                    criterion=self.criterion,
                ),
            }
            metrics = dict(target_scores, **metrics)

            fpr, tpr = metrics['roc']
            fprs.append(fpr)
            tprs.append(tpr)
            if best_auroc < metrics['auroc']:
                best_auroc = metrics['auroc']
                best_roc = metrics['roc']

            train_scores.append(metrics['train_score'])
            test_scores.append(metrics['test_score'])
            aurocs.append(metrics['auroc'])

        if config.experiments > 1:
            stats = {
                'train_acc_mean': [mean(train_scores)],
                'train_acc_stdev': [stdev(train_scores)],
                'test_acc_mean': [mean(test_scores)],
                'test_acc_stdev': [stdev(test_scores)],
                'auroc_mean': [mean(aurocs)],
                'auroc_stdev': [stdev(aurocs)],
            }
        else:
            stats = {
                'train_acc': train_scores,
                'test_acc': test_scores,
                'auroc': aurocs,
            }

        stat_df = pd.DataFrame(stats, index=[config.name])
        roc_df = pd.DataFrame({f'{config.name}_fpr': fpr, f'{config.name}_tpr': tpr}) # TODO: save fprs and tprs.
        if config.make_plots:
            prefix = f'{config.savedir}/{config.name}_roc_loglog_'
            if config.experiments > 1:
                savepath_best = prefix + 'best.png'
                savepath_multi = prefix + 'multi.png'
                fpr, tpr = best_roc
                utils.plot_roc_loglog(fpr, tpr, savepath=savepath_best) # Plot the ROC curve for sample with highest AUROC.
                utils.plot_multi_roc_loglog(fprs, tprs, savepath=savepath_multi)
            else:
                utils.plot_roc_loglog(fpr, tpr, savepath=prefix[:-1] + '.png')
        return stat_df, roc_df


def main(config):
    config['dataset'] = config['dataset'].lower()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return MembershipInferenceExperiment(config).run()

if __name__ == '__main__':
    torch.random.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="basic-shadow", type=str)
    parser.add_argument("--dataset", default="cora", type=str)
    parser.add_argument("--split", default="sampled", type=str)
    parser.add_argument("--model", default="GCN", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs-target", default=100, type=int)
    parser.add_argument("--epochs-attack", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction)
    parser.add_argument("--hidden-dim-target", default=256, type=int)
    parser.add_argument("--hidden-dim-attack", default=[128, 64], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--query-hops", default=0, type=int)
    parser.add_argument("--experiments", default=1, type=int)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--num-shadow-models", default=64, type=int)
    parser.add_argument("--rmia-offline-interp-param", default=0.6, type=float)
    parser.add_argument("--name", default="unnamed", type=str)
    parser.add_argument("--datadir", default="./data", type=str)
    parser.add_argument("--savedir", default="./results", type=str)
    args = parser.parse_args()
    config = vars(args)
    config['make_plots'] = True
    print('Running MIA experiment.')
    print(Config(config))
    print()
    stat_df, roc_df = main(config)
    print('Attack statistics:')
    print(stat_df)
    roc_df.to_csv(f'{args.savedir}/roc_{args.name}.csv', index=False)
