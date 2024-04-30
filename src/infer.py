import torch
from torch.utils.data import DataLoader
from torchmetrics import AUROC, F1Score

def test(model, dataset, criterion):
    model.eval()
    with torch.inference_mode():
        out = model(dataset.x, dataset.edge_index)
        score = criterion(out[dataset.test_mask].argmax(dim=1), dataset.y[dataset.test_mask])
    return score

def evaluate_attack_model(model, dataset):
    model.cpu()
    model.eval()
    auroc = AUROC(task='multiclass', num_classes=2)
    f1_score = F1Score(task='multiclass', num_classes=2)
    with torch.inference_mode():
        preds = model(dataset.features)
        truth = dataset.labels
        auroc_score = auroc(preds, truth).item()
        f1 = f1_score(preds, truth).item()
        print("AUROC:", auroc_score)
        print("F1:", f1)
