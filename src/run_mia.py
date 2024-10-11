import attacks
import datasetup
import hypertuner
import evaluation
import trainer
import utils

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torchmetrics import Accuracy
from statistics import mean, stdev
from collections import defaultdict
from itertools import combinations

class MembershipInferenceExperiment:

    def __init__(self, config):
        self.config = utils.Config(config)
        self.dataset = datasetup.parse_dataset(root=self.config.datadir, name=self.config.dataset)
        self.criterion = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(self.config.device)
        print(self.dataset)

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
                inductive_split=config.inductive_inference,
            ).cpu()
            dataset.to('cpu')
            hinge = utils.hinge_loss(embs, dataset.y)
            utils.plot_embedding_2D_scatter(embs=embs, y=dataset.y, train_mask=train_mask, savepath=savepath)
            for label in range(dataset.num_classes):
                label_mask = dataset.y == label
                savepath = f'{savedir}/hinge_hist_class{label}_{config.name}.png'
                utils.plot_hinge_histogram(hinge, label_mask=label_mask, train_mask=train_mask, savepath=savepath)

    def visualize_aggregation_effect_on_attack_vulnerabilities(self, attacker, target_samples, target_fpr, num_hops=2):
        soft_preds_0 = attacker.run_attack(target_samples=target_samples, num_hops=0)
        soft_preds_k = attacker.run_attack(target_samples=target_samples, num_hops=num_hops, inductive_inference=True)
        most_vul_node = torch.argmax(soft_preds_0).item()
        node_index, _, _, _ = k_hop_subgraph(
            node_idx=most_vul_node,
            num_hops=num_hops,
            edge_index=target_samples.edge_index[:, target_samples.inductive_mask],
            relabel_nodes=False,
            num_nodes=target_samples.x.shape[0],
        )
        preds_0 = soft_preds_0[node_index]
        preds_2 = soft_preds_k[node_index]
        threshold_0 = evaluation.bc_evaluation(preds=soft_preds_0, truth=target_samples.train_mask.long(), target_fpr=target_fpr)['fixed_fpr_threshold']
        threshold_k = evaluation.bc_evaluation(preds=soft_preds_k, truth=target_samples.train_mask.long(), target_fpr=target_fpr)['fixed_fpr_threshold']
        indices = torch.arange(node_index.shape[0])
        savepath = f'{self.config.savedir}/{self.config.name}_agg_vuln.png'
        plt.scatter(indices, preds_0, label='0-hop')
        plt.scatter(indices, preds_2, label=f'{num_hops}-hop', marker='x')
        plt.plot(indices, torch.ones_like(indices) * threshold_0, label=r'$\gamma$ 0-hop')
        plt.plot(indices, torch.ones_like(indices) * threshold_k, label=rf'$\gamma$ {num_hops}-hop')
        plt.xticks(np.arange(indices.shape[0]))
        plt.xlabel('Node id')
        plt.ylabel('Test statistic')
        plt.legend()
        utils.savefig_or_show(savepath)

    def train_target_model(self, dataset, plot_training_results=True):
        config = self.config

        if self.config.grid_search:
            opt_hyperparams = hypertuner.grid_search(
                dataset=dataset,
                model_type=self.config.model,
                epochs=self.config.epochs_target,
                early_stopping=self.config.early_stopping,
                optimizer=self.config.optimizer,
                transductive=config.transductive,
            )
            print(f'Grid search results: {opt_hyperparams}')
            lr, weight_decay, dropout, hidden_dim = opt_hyperparams.values()
        else:
            lr, weight_decay, dropout, hidden_dim = config.lr, config.weight_decay, config.dropout, config.hidden_dim_target

        target_model = utils.fresh_model(
            model_type=self.config.model,
            num_features=dataset.num_features,
            hidden_dims=hidden_dim,
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
        for i_experiment in range(config.experiments):
            print(f'Running experiment {i_experiment + 1}/{config.experiments}.')

            match config.attack:
                case "basic-mlp":
                    target_dataset, shadow_dataset = datasetup.target_shadow_split(dataset, split=config.split)
                    target_model = self.train_target_model(target_dataset)
                    attacker = attacks.BasicMLPAttack(
                        target_model=target_model,
                        shadow_dataset=shadow_dataset,
                        config=config,
                    )
                case "confidence":
                    target_dataset, _ = datasetup.target_shadow_split(dataset, split="disjoint")
                    target_model = self.train_target_model(target_dataset)
                    attacker = attacks.ConfidenceAttack(
                        target_model=target_model,
                        config=config,
                    )
                case "lira":
                    # In offline LiRA, the shadow models are trained on datasets that does not contain the target sample.
                    # Therefore we make a disjoint split and train shadow models on one part, and attack samples of the other part.
                    target_dataset, population = datasetup.target_shadow_split(dataset, split="disjoint")
                    target_model = self.train_target_model(target_dataset)
                    attacker = attacks.LiRA(
                        target_model=target_model,
                        population=population,
                        config=config,
                    )
                case "rmia":
                    target_dataset, population = datasetup.target_shadow_split(dataset, split="disjoint")
                    target_model = self.train_target_model(target_dataset)
                    attacker = attacks.RMIA(
                        target_model=target_model,
                        population=population,
                        config=config,
                    )
                case _:
                    raise AttributeError(f"No attack named {config.attack}")

            # Remove validation mask from target samples
            target_samples = datasetup.masked_subgraph(target_dataset, ~target_dataset.val_mask)
            truth = target_samples.train_mask.long()

            if config.experiments == 1:
                self.visualize_aggregation_effect_on_attack_vulnerabilities(attacker, target_samples, config.target_fpr, num_hops=2)

            soft_preds = []
            true_positives = []
            ids = [] # Used to label detection count dataframe
            # Run attack using the specified set of k-hop neighborhood queries.
            for num_hops in config.query_hops:
                if num_hops == 0:
                    ids.append('0')
                    preds = attacker.run_attack(target_samples=target_samples, num_hops=num_hops)
                    metrics = evaluation.bc_evaluation(preds, truth, config.target_fpr)
                    soft_preds.append(preds)
                    fpr, tpr = metrics['roc']
                    true_positives.append(metrics['TP_fixed_fpr'])
                    scores[f'fprs_{num_hops}'].append(fpr)
                    scores[f'tprs_{num_hops}'].append(tpr)
                    scores[f'auroc_{num_hops}'].append(metrics['auroc'])
                    scores[f'tprs_at_fixed_fpr_{num_hops}'].append(metrics['tpr_fixed_fpr'])
                else:
                    flags_IITI = (True, False) if config.inductive_inference is None else (config.inductive_inference,)
                    for flag in flags_IITI:
                        suffix = 'II' if flag else 'TI'
                        ids.append(f'{num_hops}-{suffix}')
                        preds = attacker.run_attack(target_samples=target_samples, num_hops=num_hops, inductive_inference=flag)
                        metrics = evaluation.bc_evaluation(preds, truth, config.target_fpr)
                        soft_preds.append(preds)
                        fpr, tpr = metrics['roc']
                        true_positives.append(metrics['TP_fixed_fpr'])
                        scores[f'fprs_{num_hops}_{suffix}'].append(fpr)
                        scores[f'tprs_{num_hops}_{suffix}'].append(tpr)
                        scores[f'auroc_{num_hops}_{suffix}'].append(metrics['auroc'])
                        scores[f'tprs_at_fixed_fpr_{num_hops}_{suffix}'].append(metrics['tpr_fixed_fpr'])

            soft_preds = torch.stack(soft_preds)
            multi_tpr, _ = utils.tpr_at_fixed_fpr_multi(soft_preds, truth, config.target_fpr)
            scores['tprs_at_fixed_fpr_multi'].append(multi_tpr)

            detection_counts = np.array(evaluation.inclusions(true_positives), dtype=np.int64)
            scores['detection_count'].append(detection_counts)

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

            scores['train_scores'].append(target_scores['train_score'])
            scores['test_scores'].append(target_scores['test_score'])

        if config.experiments > 1:
            stats = {
                'train_acc_mean': [f"{mean(scores['train_scores']):.4f}"],
                'train_acc_std': [f"{stdev(scores['train_scores']):.4f}"],
                'test_acc_mean': [f"{mean(scores['test_scores']):.4f}"],
                'test_acc_std': [f"{stdev(scores['test_scores']):.4f}"],
            }
            stats[f'tpr_{config.target_fpr:.2}_fpr_multi_mean'] = [f"{mean(scores['tprs_at_fixed_fpr_multi']):.4f}"]
            stats[f'tpr_{config.target_fpr:.2}_fpr_multi_std'] = [f"{stdev(scores['tprs_at_fixed_fpr_multi']):.4f}"]
            for num_hops in config.query_hops:
                if num_hops == 0:
                    stats[f'auroc_{num_hops}_mean'] = [f"{mean(scores[f'auroc_{num_hops}']):.4f}"]
                    stats[f'auroc_{num_hops}_std'] = [f"{stdev(scores[f'auroc_{num_hops}']):.4f}"]
                    stats[f'tpr_{config.target_fpr:.2}_fpr_{num_hops}_mean'] = [f"{mean(scores[f'tprs_at_fixed_fpr_{num_hops}']):.4f}"]
                    stats[f'tpr_{config.target_fpr:.2}_fpr_{num_hops}_std'] = [f"{stdev(scores[f'tprs_at_fixed_fpr_{num_hops}']):.4f}"]
                else:
                    for flag in flags_IITI:
                        suffix = 'II' if flag else 'TI'
                        stats[f'auroc_{num_hops}_{suffix}_mean'] = [f"{mean(scores[f'auroc_{num_hops}_{suffix}']):.4f}"]
                        stats[f'auroc_{num_hops}_{suffix}_std'] = [f"{stdev(scores[f'auroc_{num_hops}_{suffix}']):.4f}"]
                        stats[f'tpr_{config.target_fpr:.2}_fpr_{num_hops}_{suffix}_mean'] = [f"{mean(scores[f'tprs_at_fixed_fpr_{num_hops}_{suffix}']):.4f}"]
                        stats[f'tpr_{config.target_fpr:.2}_fpr_{num_hops}_{suffix}_std'] = [f"{stdev(scores[f'tprs_at_fixed_fpr_{num_hops}_{suffix}']):.4f}"]
        else: # 1 experiment
            stats = {
                'train_acc': scores['train_scores'],
                'test_acc': scores['test_scores'],
            }
            stats[f'tpr_{config.target_fpr:.2}_fpr_multi'] = scores['tprs_at_fixed_fpr_multi']
            for num_hops in config.query_hops:
                if num_hops == 0:
                    stats[f'auroc_{num_hops}'] = scores[f'auroc_{num_hops}']
                    stats[f'tpr_{config.target_fpr:.2}_fpr_{num_hops}'] = scores[f'tprs_at_fixed_fpr_{num_hops}']
                else:
                    for flag in flags_IITI:
                        suffix = 'II' if flag else 'TI'
                        stats[f'auroc_{num_hops}_{suffix}'] = scores[f'auroc_{num_hops}_{suffix}']
                        stats[f'tpr_{config.target_fpr:.2}_fpr_{num_hops}_{suffix}'] = scores[f'tprs_at_fixed_fpr_{num_hops}_{suffix}']

        detection_counts = np.stack(scores['detection_count'])
        detection_counts_mean = np.mean(detection_counts, axis=0)
        detection_counts_std = np.std(detection_counts, axis=0)
        table = {}
        i = 0
        for r in range(1, len(ids) + 1):
            for idx in combinations(range(len(ids)), r):
                key = '+'.join(ids[j] for j in idx)
                table[key] = [f'{detection_counts_mean[i]:.4f}', f'{detection_counts_std[i]:.4f}']
                i += 1

        detection_df = pd.DataFrame(table, index=['mean', 'std'])
        detection_df.to_csv('./results/detection_counts.csv')
        stat_df = pd.DataFrame(stats, index=[config.name])
        if config.make_plots:
            savepath = f'{config.savedir}/{config.name}_roc_loglog.png'
            for num_hops in config.query_hops:
                if num_hops == 0:
                    savepath = f'{config.savedir}/{config.name}_{num_hops}_roc_loglog.png'
                    fprs = scores[f'fprs_{num_hops}']
                    tprs = scores[f'tprs_{num_hops}']
                    utils.plot_multi_roc_loglog(fprs, tprs, savepath=savepath)
                else:
                    for flag in flags_IITI:
                        suffix = 'II' if flag else 'TI'
                        savepath = f'{config.savedir}/{config.name}_{num_hops}_{suffix}_roc_loglog.png'
                        fprs = scores[f'fprs_{num_hops}_{suffix}']
                        tprs = scores[f'tprs_{num_hops}_{suffix}']
                        utils.plot_multi_roc_loglog(fprs, tprs, savepath=savepath)
        return stat_df, detection_df


def main(config):
    config['dataset'] = config['dataset'].lower()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    mie = MembershipInferenceExperiment(config)
    # mie.visualize_embedding_distribution()
    return mie.run()

if __name__ == '__main__':
    torch.random.manual_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="confidence", type=str)
    parser.add_argument("--dataset", default="cora", type=str)
    parser.add_argument("--split", default="sampled", type=str)
    parser.add_argument("--transductive", action=argparse.BooleanOptionalAction)
    parser.add_argument("--inductive-inference", action=argparse.BooleanOptionalAction)
    parser.add_argument("--model", default="GCN", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs-target", default=500, type=int)
    parser.add_argument("--epochs-attack", default=100, type=int)
    parser.add_argument("--grid-search", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--early-stopping", default=30, type=int)
    parser.add_argument("--hidden-dim-target", default=[32], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--hidden-dim-attack", default=[256, 64], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--query-hops", default=[0], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--experiments", default=1, type=int)
    parser.add_argument("--target-fpr", default=0.01, type=float)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--num-shadow-models", default=64, type=int)
    parser.add_argument("--rmia-gamma", default=2.0, type=float)
    parser.add_argument("--name", default="unnamed", type=str)
    parser.add_argument("--datadir", default="./data", type=str)
    parser.add_argument("--savedir", default="./results", type=str)
    args = parser.parse_args()
    config = vars(args)
    config['make_plots'] = True
    print('Running MIA experiment.')
    print(utils.Config(config))
    print()
    stat_df, detection_df = main(config)
    print('Attack statistics:')
    print(stat_df)
    if len(detection_df.columns) > 1:
        print('Node detection count set differences:')
        print(detection_df)
