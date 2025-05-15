import attacks
import datasetup
import hypertuner
import evaluation
import trainer
import utils

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
import copy

class MembershipInferenceAudit:

    def __init__(self, config):
        config = utils.Config(config)
        self.dataset = datasetup.parse_dataset(root=config.datadir, name=config.dataset, max_num_nodes=config.max_num_nodes)
        self.dataset.to(config.device)
        self.criterion = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(config.device)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.shadow_models = None
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
                minimum_average_gen_gap=config.minimum_average_gen_gap,
            )
            print(f'Hyperparameter search results: {opt_hyperparams}')
            Path(f"results/hyperparams/{config.dataset}").mkdir(parents=True, exist_ok=True)
            log_info = f'dataset: {config.dataset}\nmodel: {config.model}\nnum_nodes: {self.dataset.num_nodes}\n' + '\n'.join(f'{k}: {v}' for k, v in opt_hyperparams.items())
            with open(f"results/hyperparams/{config.dataset}/{config.dataset}_{config.model}_{self.dataset.num_nodes}.txt", "w") as f:
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
            optimizer=torch.optim.SGD,
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

    def tune_attack_hyperparams(self):
        for attack_dict in self.config.attacks.values():
            attack_config = utils.Config(attack_dict)
            mode = 'offline' if attack_config.offline else 'online'
            name = mode + '-' + attack_config.attack
            simul_target, simul_target_train_mask = self.shadow_models[0]
            simul_graph = datasetup.remasked_graph(self.dataset, simul_target_train_mask)
            simul_shadow_models = self.shadow_models[2: 10]
            n_trials = 100
            match name:
                case 'offline-g-base':
                    hyperparam_name = 'threshold_scale_factor'
                    if hasattr(attack_config, hyperparam_name):
                        continue
                    print('Tuning threshold scale factor for offline G-BASE using optuna')
                    simul_config = copy.deepcopy(attack_config)
                    simul_config.num_sampled_graphs = 4 # Use less samples to get faster tuning
                    simul_attacker = attacks.G_BASE(
                        target_model=simul_target,
                        graph=simul_graph,
                        loss_fn=self.loss_fn,
                        config=simul_config,
                        shadow_models=simul_shadow_models,
                    )
                    n_trials = 20 # Reduce n_trails for efficiency
                case 'offline-base':
                    hyperparam_name = 'threshold_scale_factor'
                    if hasattr(attack_config, hyperparam_name):
                        continue
                    print('Tuning threshold scale factor for offline BASE using optuna')
                    simul_attacker = attacks.BASE(
                        target_model=simul_target,
                        graph=simul_graph,
                        loss_fn=self.loss_fn,
                        config=attack_config,
                        shadow_models=simul_shadow_models,
                    )
                case 'offline-rmia':
                    hyperparam_name = 'interp_param'
                    if hasattr(attack_config, hyperparam_name):
                        continue
                    print('Tuning interpolation parameter for offline RMIA using optuna')
                    simul_attacker = attacks.RMIA(
                        target_model=simul_target,
                        graph=simul_graph,
                        loss_fn=self.loss_fn,
                        config=attack_config,
                        shadow_models=simul_shadow_models,
                    )
                case _:
                    continue
            hyperparam_value = hypertuner.optuna_offline_hyperparam_tuner(
                simul_attacker,
                hyperparam_name,
                n_trials=n_trials,
            )
            attack_dict[hyperparam_name] = hyperparam_value

    def get_attacker(self, attack_dict, target_model):
        '''
        Return an instance of the attack class specified by config.attack
        '''
        attack_config = utils.Config(attack_dict)
        if self.config.pretrain_shadow_models:
            pretrained_shadow_models = self.shadow_models
        else:
            pretrained_shadow_models = None
        match attack_config.attack:
            case "g-base":
                attacker = attacks.G_BASE(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "base":
                attacker = attacks.BASE(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "laplace-base":
                attacker = attacks.LaplaceBASE(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                )
            case "b-base":
                attacker = attacks.B_BASE(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "bg-base":
                attacker = attacks.BG_BASE(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "s-base":
                attacker = attacks.S_BASE(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                )
            case "sg-base":
                attacker = attacks.SG_BASE(
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
            case "bmia":
                attacker = attacks.BMIA(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                )
            case "lira":
                attacker = attacks.LiRA(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "rmia":
                attacker = attacks.RMIA(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "mlp-attack":
                attacker = attacks.MLPAttack(
                    target_model=target_model,
                    graph=self.dataset,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case _:
                raise AttributeError(f"No attack named {attack_config.attack}")
        return attacker

    def get_target_nodes(self):
        config = self.config
        assert 0.0 <= config.frac_target_nodes <= 1.0
        assert not torch.any(self.dataset.train_mask & self.dataset.test_mask)
        num_target_nodes = int(config.frac_target_nodes * self.dataset.num_nodes)
        # Make sure the number of targets is even so there can be an equal amount of members and non-members
        if num_target_nodes % 2 == 1:
            num_target_nodes -= 1
        train_nodes = self.dataset.train_mask.long()
        test_nodes = self.dataset.test_mask.long()
        positives = train_nodes.nonzero().squeeze()
        negatives = test_nodes.nonzero().squeeze()
        num_target_nodes = min(num_target_nodes, 2 * positives.shape[0])
        perm_mask = torch.randperm(positives.shape[0])
        positives = positives[perm_mask][:num_target_nodes // 2]
        perm_mask = torch.randperm(negatives.shape[0])
        negatives = negatives[perm_mask][:num_target_nodes // 2]
        perm_mask = torch.randperm(num_target_nodes)
        target_node_index = torch.concat((positives, negatives))[perm_mask]
        assert target_node_index.shape == (num_target_nodes,)
        return target_node_index

    def parse_stats(self, stats):
        config = self.config
        frames = []
        for attack in config.attacks.keys():
            table = defaultdict(list)
            for key, value in stats[attack].items():
                if key not in ('TPR', 'FPR'):
                    table[f'{key}'].append(utils.stat_repr(value))
            frames.append(pd.DataFrame(table, index=[config.name + '_' + attack]))
        return pd.concat(frames)

    def run_audit(self):
        config = self.config
        stats = defaultdict(lambda: defaultdict(list))
        if config.pretrain_shadow_models:
            self.shadow_models = trainer.train_shadow_models(self.dataset, self.loss_fn, config)
            self.tune_attack_hyperparams()
        for i_audit in range(1, config.num_audits + 1):
            print(f'Running audit {i_audit}/{config.num_audits}')
            _ = datasetup.random_remasked_graph(self.dataset, train_frac=config.train_frac, val_frac=config.val_frac, mutate=True)
            assert not torch.any(self.dataset.val_mask), "Validation mask not fully supported"
            assert not torch.any(self.dataset.train_mask & self.dataset.test_mask)
            assert torch.all(self.dataset.train_mask | self.dataset.test_mask)
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
                for t_fpr, t_tpr, threshold in zip(config.target_fpr, metrics['TPR@FPR'], metrics['threshold@FPR']):
                    stats[attack][f'TPR@{t_fpr}FPR'].append(t_tpr)
                    stats[attack][f'threshold@{t_fpr}FPR'].append(threshold)

        stat_df = self.parse_stats(stats)
        stats = utils.nestled_defaultdict_to_dict(stats)
        return stat_df, stats

def add_attack_parameters(params):
    ''' Add target values as default values to attack config parameters. '''
    properties = [
        'model', 'epochs', 'hidden_dim', 'lr', 'weight_decay', 'optimizer', 'dropout',
        'inductive_split', 'device', 'early_stopping', 'train_frac', 'val_frac', 'batch_size',
        'hidden_dim_mlp', 'epochs_mlp', 'num_processes', 'num_shadow_models', 'offline',
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

def run(config):
    set_seed(config['seed'])
    if config['hyperparam_search']:
        MembershipInferenceAudit(config)
    else:
        add_attack_parameters(config)
        mie = MembershipInferenceAudit(config)
        return mie.run_audit()
