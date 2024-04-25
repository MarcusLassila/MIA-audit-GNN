from tqdm.auto import tqdm
from collections import defaultdict

def train_step(model, train_loader, loss_fn, optimizer, device):
    model.train()
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
    return loss.item()

def train(model, train_loader, valid_loader, loss_fn, optimizer, epochs, device):
    res = defaultdict(list)
    for _ in tqdm(range(epochs), desc=f"Training on {device}."):
        loss = train_step(model, train_loader, loss_fn, optimizer, device)
        res['loss'].append(loss)
    return res
