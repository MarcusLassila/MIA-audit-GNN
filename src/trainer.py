import torch
from tqdm.auto import tqdm
from collections import defaultdict

def train_step_gnn(model, dataset, loss_fn, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(dataset.x, dataset.edge_index)
    loss = loss_fn(out[dataset.train_mask], dataset.y[dataset.train_mask])
    score = criterion(out[dataset.train_mask].argmax(dim=1), dataset.y[dataset.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item().cpu(), score.cpu()

def valid_step_gnn(model, dataset, loss_fn, criterion):
    model.eval()
    with torch.inference_mode():
        out = model(dataset.x, dataset.edge_index)
        loss = loss_fn(out[dataset.val_mask], dataset.y[dataset.val_mask])
        score = criterion(out[dataset.val_mask].argmax(dim=1), dataset.y[dataset.val_mask])
    return loss.item().cpu(), score.cpu()

def train_gnn(model, dataset, loss_fn, optimizer, criterion, epochs, device):
    model.to(device)
    dataset.to(device)
    res = defaultdict(list)
    for _ in tqdm(range(epochs), desc=f"Training on {device}."):
        train_loss, train_score = train_step_gnn(model, dataset, loss_fn, optimizer, criterion)
        valid_loss, valid_score = valid_step_gnn(model, dataset, loss_fn, criterion)
        res['train_loss'].append(train_loss)
        res['train_score'].append(train_score)
        res['valid_loss'].append(valid_loss)
        res['valid_score'].append(valid_score)
    return res

def train_step(model, data_loader, loss_fn, optimizer, criterion, device):
    model.train()
    accumulated_loss, score = 0, 0
    for X, y in data_loader:
        optimizer.zero_grad()
        X.to(device)
        y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        accumulated_loss += loss.item()
        score += criterion(pred, y)
        loss.backward(retain_graph=True)
        optimizer.step()
    score /= len(data_loader)
    return accumulated_loss.cpu(), score.cpu()

def valid_step(model, data_loader, loss_fn, criterion, device):
    model.eval()
    accumulated_loss, score = 0, 0
    with torch.inference_mode():
        for X, y in data_loader:
            X.to(device)
            y.to(device)
            pred = model(X)
            accumulated_loss += loss_fn(pred, y).item()
            score += criterion(pred, y)
        score /= len(data_loader)
    return accumulated_loss.cpu(), score.cpu()

def train_mlp(model, train_loader, valid_loader, loss_fn, optimizer, criterion, epochs, device):
    model.to(device)
    res = defaultdict(list)
    for _ in tqdm(range(epochs), desc=f"Training MLP on {device}."):
        train_loss, train_score = train_step(model, train_loader, loss_fn, optimizer, criterion, device)
        valid_loss, valid_score = valid_step(model, valid_loader, loss_fn, criterion, device)
        res['train_loss'].append(train_loss)
        res['train_score'].append(train_score)
        res['valid_loss'].append(valid_loss)
        res['valid_score'].append(valid_score)
    return res
