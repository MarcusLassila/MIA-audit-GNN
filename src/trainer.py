import numpy as np
import torch
from tqdm.auto import tqdm
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Union

@dataclass
class TrainConfig:
    criterion: Callable[[Union[np.ndarray, torch.tensor], Union[np.ndarray, torch.tensor]], float]
    device: Union[str, torch.device]
    epochs: int
    early_stopping: int
    loss_fn: Callable[[Union[np.ndarray, torch.tensor], Union[np.ndarray, torch.tensor]], float]
    lr: float
    weight_decay: float
    optimizer: torch.optim.Optimizer

def train_step_gnn(model, dataset, optimizer, loss_fn, criterion, edge_mask=None):
    model.train()
    optimizer.zero_grad()
    if edge_mask is not None:
        out = model(dataset.x, dataset.edge_index[:, edge_mask])
    else:
        out = model(dataset.x, dataset.edge_index)
    loss = loss_fn(out[dataset.train_mask], dataset.y[dataset.train_mask])
    score = criterion(out[dataset.train_mask].argmax(dim=1), dataset.y[dataset.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item() / dataset.train_mask.sum().item(), score.item()

def valid_step_gnn(model, dataset, loss_fn, criterion, edge_mask=None):
    model.eval()
    with torch.inference_mode():
        if edge_mask is not None:
            out = model(dataset.x, dataset.edge_index[:, edge_mask])
        else:
            out = model(dataset.x, dataset.edge_index)
        loss = loss_fn(out[dataset.val_mask], dataset.y[dataset.val_mask])
        score = criterion(out[dataset.val_mask].argmax(dim=1), dataset.y[dataset.val_mask])
    return loss.item() / dataset.val_mask.sum().item(), score.item()

def train_gnn(model, dataset, config: TrainConfig, disable_tqdm=False, inductive_split=True):
    model.to(config.device)
    dataset.to(config.device)
    if inductive_split:
        edge_mask = dataset.inductive_mask
    else:
        edge_mask = dataset.random_edge_mask
    if config.early_stopping and dataset.val_mask.sum().item() == 0:
        raise Exception('Early stopping not possible without a validation set!')
    optimizer = config.optimizer(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn, criterion = config.loss_fn, config.criterion
    res = defaultdict(list)
    early_stopping_counter = 0
    min_loss = float('inf')
    best_model = None
    for _ in tqdm(range(config.epochs), disable=disable_tqdm, desc=f"Training {model.__class__.__name__} on {config.device}"):
        train_loss, train_score = train_step_gnn(model, dataset, optimizer, loss_fn, criterion, edge_mask)
        res['train_loss'].append(train_loss)
        res['train_score'].append(train_score)
        if dataset.val_mask.sum().item() > 0:
            valid_loss, valid_score = valid_step_gnn(model, dataset, loss_fn, criterion, edge_mask)
            res['valid_loss'].append(valid_loss)
            res['valid_score'].append(valid_score)
            if valid_loss < min_loss:
                min_loss = valid_loss
                best_model = deepcopy(model)
                early_stopping_counter = 0
            elif config.early_stopping > 0:
                early_stopping_counter += 1
                if early_stopping_counter == config.early_stopping:
                    break
    if config.early_stopping:
        model = best_model
    return res

def train_step(model, data_loader, optimizer, loss_fn, criterion, device):
    model.train()
    accumulated_loss, score = 0, 0
    for X, y in data_loader:
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        accumulated_loss += loss.item()
        score += criterion(logits, y).item()
        loss.backward(retain_graph=True)
        optimizer.step()
    avg_loss = accumulated_loss / len(data_loader)
    score /= len(data_loader)
    return avg_loss, score

def valid_step(model, data_loader, loss_fn, criterion, device):
    model.eval()
    accumulated_loss, score = 0, 0
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            accumulated_loss += loss_fn(logits, y).item()
            score += criterion(logits, y).item()
        avg_loss = accumulated_loss / len(data_loader)
        score /= len(data_loader)
    return avg_loss, score

def train_mlp(model, train_loader, valid_loader, config: TrainConfig):
    model.to(config.device)
    optimizer = config.optimizer(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn, criterion = config.loss_fn, config.criterion
    res = defaultdict(list)
    early_stopping_counter = 0
    min_loss = float('inf')
    best_model = None
    for _ in tqdm(range(config.epochs), desc=f"Training {model.__class__.__name__} on {config.device}"):
        train_loss, train_score = train_step(model, train_loader, optimizer, loss_fn, criterion, config.device)
        valid_loss, valid_score = valid_step(model, valid_loader, loss_fn, criterion, config.device)
        res['train_loss'].append(train_loss)
        res['train_score'].append(train_score)
        res['valid_loss'].append(valid_loss)
        res['valid_score'].append(valid_score)
        if valid_loss < min_loss:
            min_loss = valid_loss
            best_model = deepcopy(model)
            early_stopping_counter = 0
        elif config.early_stopping > 0:
            early_stopping_counter += 1
            if early_stopping_counter == config.early_stopping:
                break
    model = best_model
    return res
