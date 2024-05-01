import torch
from torch.utils.data import DataLoader
from torchmetrics import AUROC, F1Score, Precision, Recall, ROC

def test(model, dataset, criterion):
    model.eval()
    with torch.inference_mode():
        out = model(dataset.x, dataset.edge_index)
        score = criterion(out[dataset.test_mask].argmax(dim=1), dataset.y[dataset.test_mask])
    return score

def evaluate_attack_model(model, dataset):
    model.eval()
    auroc_fn = AUROC(task='multiclass', num_classes=2)
    f1_fn = F1Score(task='multiclass', num_classes=2)
    precision_fn = Precision(task='multiclass', num_classes=2)
    recall_fn = Recall(task='multiclass', num_classes=2)
    roc_fn = ROC(task='multiclass', num_classes=2)
    with torch.inference_mode():
        preds = model(dataset.features).cpu()
        truth = dataset.labels
        auroc = auroc_fn(preds, truth).item()
        f1 = f1_fn(preds, truth).item()
        precision = precision_fn(preds, truth).item()
        recall = recall_fn(preds, truth).item()
        roc = roc_fn(preds, truth)
        roc_fn.update(preds, truth)
        fig, ax = roc_fn.plot(score=True)
        fig.savefig("plots/ROC.png")
    return {
        'auroc': auroc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
    }
