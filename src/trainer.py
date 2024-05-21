import numpy as np
import torch
from tqdm.auto import tqdm
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Union

EARLY_STOPPING_THRESHOLD = 10 # Number of consecutive epochs with worse than the best seen validation set loss before early stopping.

ArrayType = Union[np.ndarray, torch.tensor]

def looper(iterable, use_tqdm, desc=""):
    return tqdm(iterable, desc=desc) if use_tqdm else iterable

@dataclass
class TrainConfig:
    criterion: Callable[[ArrayType, ArrayType], float]
    device: Union[str, torch.device]
    epochs: int
    early_stopping: bool
    loss_fn: Callable[[ArrayType, ArrayType], float]
    lr: float
    optimizer: torch.optim.Optimizer

def train_step_gnn(model, dataset, optimizer, loss_fn, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(dataset.x, dataset.edge_index)
    loss = loss_fn(out[dataset.train_mask], dataset.y[dataset.train_mask])
    score = criterion(out[dataset.train_mask].argmax(dim=1), dataset.y[dataset.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item() / dataset.train_mask.sum().item(), score.item()

def valid_step_gnn(model, dataset, loss_fn, criterion):
    model.eval()
    with torch.inference_mode():
        out = model(dataset.x, dataset.edge_index)
        loss = loss_fn(out[dataset.val_mask], dataset.y[dataset.val_mask])
        score = criterion(out[dataset.val_mask].argmax(dim=1), dataset.y[dataset.val_mask])
    return loss.item() / dataset.val_mask.sum().item(), score.item()

def train_gnn(model, dataset, config: TrainConfig, use_tqdm=True):
    model.to(config.device)
    dataset.to(config.device)
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    loss_fn, criterion = config.loss_fn, config.criterion
    res = defaultdict(list)
    early_stopping_counter = 0
    min_loss = float('inf')
    best_model = None
    for _ in looper(range(config.epochs), use_tqdm, desc=f"Training {model.__class__.__name__} on {config.device}"):
        train_loss, train_score = train_step_gnn(model, dataset, optimizer, loss_fn, criterion)
        valid_loss, valid_score = valid_step_gnn(model, dataset, loss_fn, criterion)
        res['train_loss'].append(train_loss)
        res['train_score'].append(train_score)
        res['valid_loss'].append(valid_loss)
        res['valid_score'].append(valid_score)
        if valid_loss < min_loss:
            min_loss = valid_loss
            best_model = deepcopy(model)
            early_stopping_counter = 0
        elif config.early_stopping:
            early_stopping_counter += 1
            if early_stopping_counter == EARLY_STOPPING_THRESHOLD:
                break
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
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
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
        elif config.early_stopping:
            early_stopping_counter += 1
            if early_stopping_counter == EARLY_STOPPING_THRESHOLD:
                break
    model = best_model
    return res
