import torch
from torch_geometric.utils import k_hop_subgraph
from torchmetrics import AUROC, F1Score, Precision, Recall, ROC

def evaluate_graph_model(model, dataset, mask, criterion):
    model.eval()
    with torch.inference_mode():
        out = model(dataset.x, dataset.edge_index)
        score = criterion(out[mask].argmax(dim=1), dataset.y[mask])
    return score.item()

def evaluate_attack_model(attack_model, target_model, dataset, num_hops=0):
    attack_model.eval()
    target_model.eval()
    device = torch.device(next(attack_model.parameters()).device)
    auroc_fn = AUROC(task='binary').to(device)
    f1_fn = F1Score(task='binary').to(device)
    precision_fn = Precision(task='binary').to(device)
    recall_fn = Recall(task='binary').to(device)
    roc_fn = ROC(task='binary').to(device)
    with torch.inference_mode():
        features = []
        for v in range(dataset.x.shape[0]):
            node_index, edge_index, v_idx, _ = k_hop_subgraph(
                node_idx=v,
                num_hops=num_hops,
                edge_index=dataset.edge_index,
                relabel_nodes=True
            )
            pred = target_model(dataset.x[node_index], edge_index)[v_idx]
            features.append(pred)
        features = torch.cat(features, dim=0)
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
