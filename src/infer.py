import utils

import torch
from torchmetrics import AUROC, F1Score, Precision, Recall, ROC
from pathlib import Path

def evaluate_graph_model(model, dataset, mask, criterion):
    model.eval()
    with torch.inference_mode():
        out = model(dataset.x, dataset.edge_index)
        score = criterion(out[mask].argmax(dim=1), dataset.y[mask])
    return score

def evaluate_attack_model(model, dataset, device):
    model.eval()
    auroc_fn = AUROC(task='binary').to(device)
    f1_fn = F1Score(task='binary').to(device)
    precision_fn = Precision(task='binary').to(device)
    recall_fn = Recall(task='binary').to(device)
    roc_fn = ROC(task='binary').to(device)
    with torch.inference_mode():
        logits = model(dataset.features)[:,1]
        truth = dataset.labels
        auroc = auroc_fn(logits, truth).item()
        f1 = f1_fn(logits, truth).item()
        precision = precision_fn(logits, truth).item()
        recall = recall_fn(logits, truth).item()
        fpr, tpr, _ = roc_fn(logits, truth)
    return {
        'auroc': auroc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc': (fpr, tpr)
    }
