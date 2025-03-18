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
from statistics import mean

def grid_search(
    param_grid: dict,
    dataset: Data,
    model_type: str,
    optimizer: str,
    inductive_split: bool,
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
        if average_valid_loss < min_valid_loss:
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
