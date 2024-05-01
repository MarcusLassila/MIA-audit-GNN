import torch
from torchmetrics import AUROC, F1Score, Precision, Recall, ROC
from pathlib import Path

def test(model, dataset, criterion):
    model.eval()
    with torch.inference_mode():
        out = model(dataset.x, dataset.edge_index)
        score = criterion(out[dataset.test_mask].argmax(dim=1), dataset.y[dataset.test_mask])
    return score

def evaluate_attack_model(model, dataset, device, savedir):
    model.eval()
    auroc_fn = AUROC(task='multiclass', num_classes=2).to(device)
    f1_fn = F1Score(task='multiclass', num_classes=2).to(device)
    precision_fn = Precision(task='multiclass', num_classes=2).to(device)
    recall_fn = Recall(task='multiclass', num_classes=2).to(device)
    roc_fn = ROC(task='multiclass', num_classes=2).to(device)
    with torch.inference_mode():
        preds = model(dataset.features)
        truth = dataset.labels
        auroc = auroc_fn(preds, truth).item()
        f1 = f1_fn(preds, truth).item()
        precision = precision_fn(preds, truth).item()
        recall = recall_fn(preds, truth).item()
        roc = roc_fn(preds, truth)
        roc_fn.update(preds, truth)
        fig, ax = roc_fn.plot(score=True)
        Path(savedir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{savedir}/ROC.png")
    return {
        'auroc': auroc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
    }
