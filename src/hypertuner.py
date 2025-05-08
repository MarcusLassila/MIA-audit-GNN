import datasetup
import trainer
import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from itertools import product
from statistics import mean
import optuna

def grid_search(
    param_grid: dict,
    dataset: Data,
    model_type: str,
    optimizer: str,
    inductive_split: bool,
    minimum_average_gen_gap: float,
    num_samples: int = 10,
):
    assert torch.any(dataset.val_mask)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(device)
    desc=f'Running grid search over the following hyperparameters: {", ".join(param_grid.keys())}'
    opt_hyperparams = None
    min_valid_loss = float('inf')
    final_performance = {}
    for lr, weight_decay, dropout, hidden_dim, epochs in tqdm(list(product(*param_grid.values())), desc=desc):
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=device,
            epochs=epochs,
            early_stopping=0,
            loss_fn=F.cross_entropy,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=getattr(torch.optim, optimizer),
        )
        valid_losses = []
        valid_scores = []
        train_scores = []
        for _ in range(num_samples):
            model = utils.fresh_model(
                model_type=model_type,
                num_features=dataset.num_features,
                hidden_dims=hidden_dim,
                num_classes=dataset.num_classes,
                dropout=dropout,
            )
            train_res = trainer.train_gnn(
                model=model,
                dataset=dataset,
                config=train_config,
                disable_tqdm=True,
                inductive_split=inductive_split,
            )
            valid_losses.append(train_res['valid_loss'][-1])
            valid_scores.append(train_res['valid_score'][-1])
            train_scores.append(train_res['train_score'][-1])
        average_valid_loss = mean(valid_losses)
        average_valid_score = mean(valid_scores)
        average_gen_gap = mean(train_scores) - average_valid_score
        if average_valid_loss < min_valid_loss and average_gen_gap > minimum_average_gen_gap:
            min_valid_loss = average_valid_loss
            opt_hyperparams = {
                'lr': lr,
                'weight_decay': weight_decay,
                'dropout': dropout,
                'hidden_dim': hidden_dim,
                'epochs': epochs,
            }
            final_performance = {
                'gen_gap': average_gen_gap,
                'valid_score': average_valid_score,
            }
    print('Final validation results:', final_performance)
    return opt_hyperparams

def optuna_offline_hyperparam_tuner(attacker, hyperparam_attr_name, n_trials=100, execute_silently=False):
    def objective(trial):
        hyperparam_value = trial.suggest_float(hyperparam_attr_name, 0.0, 1.0)
        setattr(attacker, hyperparam_attr_name, hyperparam_value)
        target_node_index = torch.arange(attacker.graph.num_nodes).to(attacker.config.device)
        score = attacker.run_attack(target_node_index).cpu().numpy()
        truth = attacker.graph.train_mask.long().cpu().numpy()
        auroc = roc_auc_score(y_true=truth, y_score=score)
        return auroc

    study = optuna.create_study(direction='maximize')
    if execute_silently:
        utils.execute_silently(study.optimize, objective, n_trials=n_trials, show_progress_bar=True)
    else:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best optimized value: {study.best_value}")
    return study.best_params[hyperparam_attr_name]
