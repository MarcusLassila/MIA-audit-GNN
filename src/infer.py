import torch

def test(model, data_loader, criterion, device):
    model.eval()
    score = 0
    with torch.inference_mode():
        for batch in data_loader:
            batch.to(device)
            out = model(batch.x, batch.edge_index)
            score += criterion(out[batch.test_mask].argmax(dim=1), batch.y[batch.test_mask])
        score /= len(data_loader)
    return score
