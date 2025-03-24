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
from tqdm.auto import tqdm
from collections import defaultdict
from pathlib import Path

class MembershipInferenceAudit:

    def __init__(self, config):
        config = utils.Config(config)
        self.dataset = datasetup.parse_dataset(root=config.datadir, name=config.dataset, max_num_nodes=config.max_num_nodes)
        self.criterion = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(config.device)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.shadow_models = None
        self.shadow_train_masks = None
        print(utils.graph_info(self.dataset))
        if config.hyperparam_search:
            val_frac = config.val_frac or config.train_frac
            _ = datasetup.random_remasked_graph(self.dataset, train_frac=config.train_frac, val_frac=val_frac, mutate=True)
            opt_hyperparams = hypertuner.grid_search(
                param_grid=config.hyperparam_grid,
                dataset=self.dataset,
                model_type=config.model,
                optimizer=config.optimizer,
                inductive_split=config.inductive_split,
            )
            print(f'Hyperparameter search results: {opt_hyperparams}')
            Path("results/hyperparams").mkdir(parents=True, exist_ok=True)
            log_info = f'dataset: {config.dataset}\nmodel: {config.model}\nnum_nodes: {self.dataset.num_nodes}\n' + '\n'.join(f'{k}: {v}' for k, v in opt_hyperparams.items())
            with open(f"results/hyperparams/{config.dataset}_{self.dataset.num_nodes}.txt", "w") as f:
                f.write(log_info)
        self.config = config

    def train_target_model(self, dataset):
        config = self.config
        target_model = utils.fresh_model(
            model_type=config.model,
            num_features=dataset.num_features,
            hidden_dims=config.hidden_dim,
            num_classes=dataset.num_classes,
            dropout=config.dropout,
        )
        train_config = trainer.TrainConfig(
            criterion=self.criterion,
            device=config.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        print(f'Training a {config.model} target model on {config.dataset}...')
        _ = trainer.train_gnn(
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
        )
        return target_model

    def train_shadow_models(self):
        config = self.config
        self.shadow_models = []
        self.shadow_train_masks = []
        criterion = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        shadow_train_masks = utils.partition_training_sets(num_nodes=self.dataset.num_nodes, num_models=config.num_shadow_models)
        for shadow_train_mask in tqdm(shadow_train_masks, total=shadow_train_masks.shape[0], desc=f"Training {config.num_shadow_models} shadow models"):
            shadow_dataset = datasetup.remasked_graph(self.dataset, shadow_train_mask)
            shadow_model = utils.fresh_model(
                model_type=config.model,
                num_features=shadow_dataset.num_features,
                hidden_dims=config.hidden_dim,
                num_classes=shadow_dataset.num_classes,
                dropout=config.dropout,
            )
            _ = trainer.train_gnn(
                model=shadow_model,
                dataset=shadow_dataset,
                config=train_config,
                disable_tqdm=True,
                inductive_split=config.inductive_split,
            )
            shadow_model.eval()
            self.shadow_models.append(shadow_model)
            self.shadow_train_masks.append(shadow_train_mask)

    def get_attacker(self, attack_dict, target_model):
        '''
        Return an instance of the attack class specified by config.attack
        '''
        attack_config = utils.Config(attack_dict)
        match attack_config.attack:
            case "bayes-optimal":
                attacker = attacks.BayesOptimalMembershipInference(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=self.shadow_models,
                )
            case "lset":
                attacker = attacks.LSET(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=self.shadow_models,
                )
            case "improved-lset":
                attacker = attacks.ImprovedLSET(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=self.shadow_models,
                )
            case "graph-lset":
                attacker = attacks.GraphLSET(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=self.shadow_models,
                )
            case "strong-lset":
                attacker = attacks.StrongLSET(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                )
            case "strong-graph-lset":
                attacker = attacks.StrongGraphLSET(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                )
            case "confidence":
                attacker = attacks.ConfidenceAttack(
                    target_model=target_model,
                    graph=self.dataset,
                    config=attack_config,
                )
            case "lira":
                attacker = attacks.LiraOnline(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=self.shadow_models,
                    shadow_train_masks=self.shadow_train_masks,
                )
            case "rmia":
                attacker = attacks.RmiaOnline(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=self.shadow_models,
                )
            case "mlp-attack":
                attacker = attacks.MLPAttack(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                )
            case _:
                raise AttributeError(f"No attack named {attack_config.attack}")
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
        frames = []
        for attack in config.attacks.keys():
            table = defaultdict(list)
            for key, value in stats[attack].items():
                if isinstance(value[0], float):
                    table[f'{key}'].append(utils.stat_repr(value))
            frames.append(pd.DataFrame(table, index=[config.name + '_' + attack]))
        return pd.concat(frames)

    def run_audit(self):
        config = self.config
        stats = defaultdict(lambda: defaultdict(list))
        if config.pretrain_shadow_models:
            self.train_shadow_models()
        for i_audit in range(1, config.num_audits + 1):
            print(f'Running audit {i_audit}/{config.num_audits}')
            _ = datasetup.random_remasked_graph(self.dataset, train_frac=config.train_frac, val_frac=config.val_frac, mutate=True)
            target_node_index = self.get_target_nodes()
            target_model = self.train_target_model(self.dataset)
            target_model.eval()
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
            for attack, attack_dict in config.attacks.items():
                attacker = self.get_attacker(attack_dict, target_model)
                stats[attack]['train_acc'].append(target_scores['train_acc'])
                stats[attack]['test_acc'].append(target_scores['test_acc'])
                truth = self.dataset.train_mask.long()[target_node_index]
                preds = attacker.run_attack(target_node_index=target_node_index)
                metrics = evaluation.evaluate_binary_classification(preds, truth, config.target_fpr, target_node_index, self.dataset)
                fpr, tpr = metrics['ROC']
                stats[attack]['FPR'].append(fpr)
                stats[attack]['TPR'].append(tpr)
                stats[attack]['AUC'].append(metrics['AUC'])
                for t_fpr, t_tpr in zip(config.target_fpr, metrics['TPR@FPR']):
                    stats[attack][f'TPR@{t_fpr}FPR'].append(t_tpr)

        stat_df = self.parse_stats(stats)
        roc_df = pd.DataFrame({})
        roc_frames = []
        for attack in config.attacks.keys():
            for i in range(config.num_audits):
                roc_frames.append(pd.DataFrame({
                    f'FPR_{i}_{config.name}_{attack}': stats[attack]['FPR'][i],
                    f'TPR_{i}_{config.name}_{attack}': stats[attack]['TPR'][i],
                }))
        roc_df = pd.concat(roc_frames)
        return stat_df, roc_df

def add_attack_parameters(params):
    ''' Add target values as default values to attack config parameters. '''
    properties = [
        'model', 'epochs', 'hidden_dim', 'lr', 'weight_decay', 'optimizer', 'dropout',
        'inductive_split', 'device', 'early_stopping', 'train_frac', 'val_frac', 'batch_size',
        'hidden_dim_mlp', 'epochs_mlp', 'num_processes',
    ]
    for attack_params in params['attacks'].values():
        for prop in properties:
            if prop not in attack_params:
                attack_params[prop] = params[prop]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(config):
    set_seed(config['seed'])
    if config['hyperparam_search']:
        MembershipInferenceAudit(config)
    else:
        add_attack_parameters(config)
        mie = MembershipInferenceAudit(config)
        return mie.run_audit()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="confidence", type=str)
    parser.add_argument("--dataset", default="cora", type=str)
    parser.add_argument("--inductive-split", action=argparse.BooleanOptionalAction)
    parser.add_argument("--inductive-inference", action=argparse.BooleanOptionalAction)
    parser.add_argument("--pretrain_shadow_models", action=argparse.BooleanOptionalAction)
    parser.add_argument("--model", default="GCN", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--epochs-mlp", default=500, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--early-stopping", default=0, type=int)
    parser.add_argument("--hidden-dim", default=[32], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--hidden-dim-mlp", default=[128], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--num-audits", default=1, type=int)
    parser.add_argument("--target-fpr", default=[0.01], type=lambda x: [*map(float, x.split(','))])
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--mlp-attack-queries", default=[0], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--bayes-sampling-strategy", default='model-independent', type=str)
    parser.add_argument("--num-shadow-models", default=10, type=int)
    parser.add_argument("--num-sampled-graphs", default=10, type=int)
    parser.add_argument("--num-target-nodes", default=500, type=int)
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
    config['name'] = config['dataset'] + '_' + config['model']
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['hyperparam_search'] = False
    if config['inductive_split'] is None:
        config['inductive_split'] = True
    if config['inductive_inference'] is None:
        config['inductive_inference'] = True
    # Construct attack config.
    # Add all default properties even if they are not applicable to the specified attack.
    config['attacks'] = {
        config['attack']: {
            'attack': config['attack'],
            'num_shadow_models': config['num_shadow_models'],
            'num_sampled_graphs': config['num_sampled_graphs'],
            'bayes_sampling_strategy': config['bayes_sampling_strategy'],
            'mlp_attack_queries': list(config['mlp_attack_queries']),
        }
    }
    del config['attack']
    print('Running MIA audit...')
    print(utils.Config(config))
    print()
    stat_df, _ = main(config)
    pd.set_option('display.max_columns', 500)
    print('Results:')
    print(stat_df)
