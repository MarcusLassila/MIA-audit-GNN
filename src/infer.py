import torch
from torchmetrics import AUROC, F1Score, Precision, Recall, ROC

def evaluate_graph_model(model, dataset, mask, criterion):
    model.eval()
    with torch.inference_mode():
        out = model(dataset.x, dataset.edge_index)
        score = criterion(out[mask].argmax(dim=1), dataset.y[mask])
    return score.item()

def evaluate_attack_model(attack_model, target_model, dataset, device):
    # TODO: Need to support k-kop queries rather than self-loop queries.
    attack_model.eval()
    target_model.eval()
    auroc_fn = AUROC(task='binary').to(device)
    f1_fn = F1Score(task='binary').to(device)
    precision_fn = Precision(task='binary').to(device)
    recall_fn = Recall(task='binary').to(device)
    roc_fn = ROC(task='binary').to(device)
    with torch.inference_mode():
        features = []
        for v in range(dataset.x.shape[0]): # TODO: Evaluate only on target subset
            features.append(target_model(dataset.x[v].unsqueeze(dim=0), torch.tensor([[v], [v]])).squeeze()) # Only self-loop
        features = torch.stack(features, dim=0)
        logits = attack_model(features)[:,1]
        truth = dataset.train_mask.long()
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
