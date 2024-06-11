import trainer
import utils

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from itertools import product

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
