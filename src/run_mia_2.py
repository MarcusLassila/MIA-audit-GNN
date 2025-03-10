import attacks
import datasetup
import hypertuner
import evaluation
import trainer
import utils

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchmetrics import Accuracy
from collections import defaultdict
from pathlib import Path

class MembershipInferenceExperiment:

    def __init__(self, config):
        config = utils.Config(config)
        self.dataset = datasetup.parse_dataset(root=config.datadir, name=config.dataset, max_num_nodes=config.max_num_nodes)
        self.criterion = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(config.device)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        print(utils.graph_info(self.dataset))
        if config.hyperparam_search:
            val_frac = config.val_frac or config.train_frac
            datasetup.add_masks(self.dataset, train_frac=config.train_frac, val_frac=val_frac)
            opt_hyperparams = hypertuner.grid_search(
                dataset=self.dataset,
                model_type=config.model,
                optimizer=config.optimizer,
                inductive_split=config.inductive_split,
            )
            print(f'Hyperparameter search results: {opt_hyperparams}')
            Path("results/hyperparams").mkdir(parents=True, exist_ok=True)
            log_info = f'dataset: {config.dataset}\nnum_nodes: {self.dataset.num_nodes}\n' + '\n'.join(f'{k}: {v}' for k, v in opt_hyperparams.items())
            with open(f"results/hyperparams/{config.dataset}_{self.dataset.num_nodes}.txt", "w") as f:
                f.write(log_info)
            print('Updating hyperparameter values accordingly')
            config.lr, config.weight_decay, config.dropout, config.hidden_dim_target, config.epochs_target = opt_hyperparams.values()
        self.config = config

    def train_target_model(self, dataset, plot_training_results=True):
        config = self.config
        target_model = utils.fresh_model(
            model_type=self.config.model,
            num_features=dataset.num_features,
            hidden_dims=config.hidden_dim_target,
            num_classes=dataset.num_classes,
            dropout=config.dropout,
        )
        train_config = trainer.TrainConfig(
            criterion=self.criterion,
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        train_res = trainer.train_gnn(
            model=target_model,
            dataset=dataset,
            config=train_config,
            inductive_split=config.inductive_split
        )
        evaluation.evaluate_graph_training(
            model=target_model,
            dataset=dataset,
            criterion=train_config.criterion,
            inductive_inference=config.inductive_split,
            training_results=train_res if plot_training_results else None,
            plot_name="target_model",
            savedir=config.savedir,
        )
        return target_model

    def get_attacker(self, target_model):
        '''
        Return an instance of the attack class specified by config.attack
        '''
        config = self.config
        match config.attack:
            case "bayes-optimal":
                attacker = attacks.BayesOptimalMembershipInference(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=config,
                )
            case "lset":
                attacker = attacks.LSET(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=config,
                )
            case "graph-lset":
                attacker = attacks.GraphLSET(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=config,
                )
            case "bootstrapped-lset":
                attacker = attacks.BootstrappedLSET(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=config,
                )
            case "confidence":
                attacker = attacks.ConfidenceAttack2(
                    target_model=target_model,
                    graph=self.dataset,
                    config=config,
                )
            case "lira":
                attacker = attacks.LiraOnline(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=config,
                )
            case "rmia":
                attacker = attacks.RmiaOnline(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=config,
                )
            case _:
                raise AttributeError(f"No attack named {config.attack}")
        return attacker

    def get_target_nodes(self):
        config = self.config
        if config.num_target_nodes == -1:
            target_node_index = torch.arange(self.dataset.num_nodes)
        else:
            assert config.num_target_nodes <= self.dataset.num_nodes
            truth = self.dataset.train_mask.long()
            num_targets = config.num_target_nodes
            positives = truth.nonzero().squeeze()
            negatives = (truth ^ 1).nonzero().squeeze()
            perm_mask = torch.randperm(positives.shape[0])
            positives = positives[perm_mask][:num_targets // 2]
            perm_mask = torch.randperm(negatives.shape[0])
            negatives = negatives[perm_mask][:num_targets // 2]
            perm_mask = torch.randperm(num_targets)
            target_node_index = torch.concat((positives, negatives))[perm_mask]
            assert target_node_index.shape == (num_targets,)
        return target_node_index

    def parse_stats(self, stats):
        config = self.config
        table = defaultdict(list)
        for key, value in stats.items():
            if isinstance(value[0], float):
                table[f'{key}'].append(utils.stat_repr(value))
        return pd.DataFrame(table, index=[config.name])

    def run_experiment(self):
        config = self.config
        stats = defaultdict(list)

        for i_experiment in range(1, config.num_experiments + 1):
            print(f'Running experiment {i_experiment}/{config.num_experiments}')

            # Use fixed random seeds such that each experimental configuration is evaluated on the same dataset
            set_seed(config.seed + i_experiment)

            datasetup.add_masks(self.dataset, train_frac=config.train_frac, val_frac=config.val_frac)
            target_node_index = self.get_target_nodes()
            target_model = self.train_target_model(self.dataset)
            attacker = self.get_attacker(target_model)
            target_scores = {
                'train_acc': evaluation.evaluate_graph_model(
                    model=target_model,
                    dataset=self.dataset,
                    mask=self.dataset.train_mask,
                    criterion=self.criterion,
                    inductive_inference=config.inductive_split,
                ),
                'test_acc': evaluation.evaluate_graph_model(
                    model=target_model,
                    dataset=self.dataset,
                    mask=self.dataset.test_mask,
                    criterion=self.criterion,
                    inductive_inference=config.inductive_split,
                ),
            }
            stats['train_acc'].append(target_scores['train_acc'])
            stats['test_acc'].append(target_scores['test_acc'])

            truth = self.dataset.train_mask.long()[target_node_index]
            preds = attacker.run_attack(target_node_index=target_node_index)
            metrics = evaluation.evaluate_binary_classification(preds, truth, config.target_fpr)
            fpr, tpr = metrics['ROC']
            stats['FPR'].append(fpr)
            stats['TPR'].append(tpr)
            stats['AUC'].append(metrics['AUC'])
            stats[f'TPR@{config.target_fpr}FPR'].append(metrics['TPR@FPR'])

        if config.make_roc_plots:
            savepath = f'{config.savedir}/{config.name}_roc_loglog.png'
            utils.plot_multi_roc_loglog(stats['FPR'], stats['TPR'], savepath=savepath)

        stats_df = self.parse_stats(stats)
        roc_df = pd.DataFrame({
            f'FPR_{config.name}': stats['FPR'][0],
            f'TPR_{config.name}': stats['TPR'][0],
        })
        return stats_df, roc_df

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def main(config):
    set_seed(config['seed'])
    config['dataset'] = config['dataset'].lower()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    mie = MembershipInferenceExperiment(config)
    return mie.run_experiment()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="confidence", type=str)
    parser.add_argument("--dataset", default="cora", type=str)
    parser.add_argument("--inductive-split", action=argparse.BooleanOptionalAction)
    parser.add_argument("--inductive-inference", action=argparse.BooleanOptionalAction)
    parser.add_argument("--model", default="GCN", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs-target", default=15, type=int)
    parser.add_argument("--epochs-shadow", default=15, type=int)
    parser.add_argument("--hyperparam-search", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--early-stopping", default=0, type=int)
    parser.add_argument("--hidden-dim-target", default=[32], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--num-experiments", default=1, type=int)
    parser.add_argument("--target-fpr", default=0.01, type=float)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--bayes-sampling-strategy", default='model-independent', type=str)
    parser.add_argument("--num-shadow-models", default=10, type=int)
    parser.add_argument("--num-sampled-graphs", default=10, type=int)
    parser.add_argument("--num-target-nodes", default=-1, type=int)
    parser.add_argument("--max-num-nodes", default=None, type=int)
    parser.add_argument("--rmia-gamma", default=2.0, type=float)
    parser.add_argument("--name", default="unnamed", type=str)
    parser.add_argument("--datadir", default="./data", type=str)
    parser.add_argument("--savedir", default="./results", type=str)
    parser.add_argument("--num-processes", default=1, type=int)
    parser.add_argument("--train-frac", default=0.5, type=float)
    parser.add_argument("--val-frac", default=0.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    config = vars(args)
    config['make_roc_plots'] = True
    config['name'] = config['dataset'] + '_' + config['model'] + '_' + config['attack']
    if config['inductive_split'] is None:
        config['inductive_split'] = True
    if config['inductive_inference'] is None:
        config['inductive_inference'] = True
    print('Running MIA experiment v2.')
    print(utils.Config(config))
    print()
    stat_df, _ = main(config)
    pd.set_option('display.max_columns', 500)
    print('Results:')
    print(stat_df)
