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
from statistics import mean

def grid_search(
    dataset: Data,
    model_type: str,
    optimizer: str,
    inductive_split: bool,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(device)
    grid = {
        'lr': [1e-2],
        'weight_decay': [1e-5, 1e-4],
        'dropout': [0.0, 0.5],
        'hidden_dim': [[32], [128], [512]],
        'epochs': [15, 100, 200, 500]
    }
    desc=f'Running grid search over the following hyperparameters: {", ".join(grid.keys())}'
    opt_hyperparams = None
    min_valid_loss = float('inf')
    for lr, weight_decay, dropout, hidden_dim, epochs in tqdm(list(product(*grid.values())), desc=desc):
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
        for _ in range(3):
            model = utils.fresh_model(
                model_type=model_type,
                num_features=dataset.num_features,
                hidden_dims=hidden_dim,
                num_classes=dataset.num_classes,
                dropout=dropout,
            )
            valid_loss = min(trainer.train_gnn(
                model=model,
                dataset=dataset,
                config=train_config,
                disable_tqdm=True,
                inductive_split=inductive_split,
            )['valid_loss'])
            valid_losses.append(valid_loss)
        average_valid_loss = mean(valid_losses)
        if average_valid_loss < min_valid_loss:
            min_valid_loss = average_valid_loss
            opt_hyperparams = {
                'lr': lr,
                'weight_decay': weight_decay,
                'dropout': dropout,
                'hidden_dim': hidden_dim,
                'epochs': epochs
            }
    return opt_hyperparams
