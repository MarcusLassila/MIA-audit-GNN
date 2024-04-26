import torch
from tqdm.auto import tqdm
from collections import defaultdict

def train_step(model, data_loader, loss_fn, optimizer, criterion, device):
    model.train()
    avg_loss, avg_score = 0, 0
    for batch in data_loader:
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
        avg_loss += loss.item()
        avg_score += criterion(out[batch.train_mask].argmax(dim=1), batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
    avg_loss /= len(data_loader)
    avg_score /= len(data_loader)
    return avg_loss, avg_score

def valid_step(model, data_loader, loss_fn, criterion, device):
    model.eval()
    avg_loss, avg_score = 0, 0
    with torch.inference_mode():
        for batch in data_loader:
            batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = loss_fn(out[batch.val_mask], batch.y[batch.val_mask])
            avg_loss += loss
            avg_score += criterion(out[batch.val_mask].argmax(dim=1), batch.y[batch.val_mask])
        avg_loss /= len(data_loader)
        avg_score /= len(data_loader)
    return avg_loss, avg_score

def train(model, data_loader, loss_fn, optimizer, criterion, epochs, device):
    res = defaultdict(list)
    for _ in tqdm(range(epochs), desc=f"Training on {device}."):
        train_loss, train_score = train_step(model, data_loader, loss_fn, optimizer, criterion, device)
        res['train_loss'].append(train_loss)
        res['train_score'].append(train_score)
        valid_loss, valid_score = valid_step(model, data_loader, loss_fn, criterion, device)
        res['valid_loss'].append(valid_loss)
        res['valid_score'].append(valid_score)
    return res
