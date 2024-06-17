import attacks
import datasetup
import trainer
import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from itertools import product
import matplotlib.pyplot as plt

def grid_search(
    dataset: Data,
    model_type: str,
    epochs: int,
    early_stopping: bool,
    optimizer: str,
    hidden_dim: int,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(device)
    grid = {
        'lr': [5e-4, 1e-3, 5e-3, 1e-2],
        'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4],
        'dropout': [0.0, 0.25, 0.5],
    }
    desc=f'Running grid search over the following hyperparameters: {", ".join(grid.keys())}'
    opt_hyperparams = None
    min_valid_loss = float('inf')
    for lr, weight_decay, dropout in tqdm(product(*grid.values()), desc=desc):
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=device,
            epochs=epochs,
            early_stopping=early_stopping,
            loss_fn=F.cross_entropy,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=getattr(torch.optim, optimizer),
        )
        model = utils.fresh_model(
            model_type=model_type,
            num_features=dataset.num_features,
            hidden_dim=hidden_dim,
            num_classes=dataset.num_classes,
            dropout=dropout,
        )
        valid_loss = min(trainer.train_gnn(
            model=model,
            dataset=dataset,
            config=train_config,
            use_tqdm=False,
        )['valid_loss'])
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            opt_hyperparams = {
                'lr': lr,
                'weight_decay': weight_decay,
                'dropout': dropout
            }

    return opt_hyperparams

def rmia_offline_interp_param_search(
    dataset: str,
    model_type: str,
    datadir: str,
):
    dataset = datasetup.parse_dataset(root=datadir, name=dataset)
    target_samples, population = datasetup.target_shadow_split(dataset=dataset, split='disjoint', target_frac=0.5, shadow_frac=0.5)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = utils.Config({
        'device': device,
        'dataset': dataset,
        'datadir': datadir,
        'model': model_type,
        'epochs_target': 500,
        'dropout': 0.5,
        'early_stopping': True,
        'lr': 1e-2,
        'weight_decay': 1e-4,
        'optimizer': 'Adam',
        'hidden_dim_target': 32,
        'query_hops': 0,
        'num_shadow_models': 8,
    })
    target_model = utils.fresh_model(
        model_type=model_type,
        num_features=dataset.num_features,
        hidden_dim=config.hidden_dim_target,
        num_classes=dataset.num_classes,
        dropout=config.dropout,
    )
    criterion = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(device)
    train_config = trainer.TrainConfig(
        criterion=criterion,
        device=device,
        epochs=config.epochs_target,
        early_stopping=config.early_stopping,
        loss_fn=F.cross_entropy,
        lr=config.lr,
        weight_decay=config.weight_decay,
        optimizer=torch.optim.Adam,
    )
    _ = trainer.train_gnn(
        model=target_model,
        dataset=target_samples,
        config=train_config,
    )

    best_auroc = 0.0
    opt_interp_param = 0.0
    aurocs = []
    param_pool = np.arange(0.0, 1.0, 0.1)
    for interp_param in param_pool:
        config.rmia_offline_interp_param = interp_param
        auroc = attacks.RMIA(
            target_model=target_model,
            population=population,
            config=config,
        ).run_attack(target_samples=target_samples)['auroc']
        aurocs.append(auroc)
        if auroc > best_auroc:
            best_auroc = auroc
            opt_interp_param = interp_param
    plt.plot(param_pool, aurocs)
    plt.show()
    return opt_interp_param

if __name__ == '__main__':
    interp_param = rmia_offline_interp_param_search(
        dataset='cora',
        model_type='GCN',
        datadir='./data',
    )
    print(f"RMIA offline interpolation parameter: {interp_param}")
