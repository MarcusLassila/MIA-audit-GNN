import torch
from torch_geometric.loader import DataLoader

def test(model, dataset, criterion):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    score = 0
    with torch.inference_mode():
        for batch in data_loader:
            out = model(batch.x, batch.edge_index)
            score = criterion(out[batch.test_mask].argmax(dim=1), batch.y[batch.test_mask])
        score /= len(data_loader)
    return score
