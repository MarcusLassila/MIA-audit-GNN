import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torchmetrics import AUROC, F1Score, Precision, Recall, ROC

def bc_evaluation(preds, labels, device, threshold=0.5):
    auroc = AUROC(task='binary').to(device)(preds, labels).item()
    f1 = F1Score(task='binary', threshold=threshold).to(device)(preds, labels).item()
    precision = Precision(task='binary', threshold=threshold).to(device)(preds, labels).item()
    recall = Recall(task='binary', threshold=threshold).to(device)(preds, labels).item()
    fpr, tpr, _ = ROC(task='binary').to(device)(preds, labels)
    return {
        'auroc': auroc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc': (fpr, tpr),
    }

def query_attack_features(model, dataset, query_nodes, num_hops=0):
    '''
    Queries the model for each node in in query_nodes,
    using the local subgraph definded by the "num_hops"-hop neigborhood.

    Output: Matrix of size "number of query nodes" times "number of classes",
            consisting of logits/predictions for each query node.
    '''
    model.eval()
    features = []
    for v in query_nodes:
        node_index, edge_index, v_idx, _ = k_hop_subgraph(
            node_idx=v,
            num_hops=num_hops,
            edge_index=dataset.edge_index,
            relabel_nodes=True,
            num_nodes=dataset.x.shape[0],
        )
        pred = model(dataset.x[node_index], edge_index)[v_idx].squeeze()
        features.append(pred)
    return torch.stack(features)

def evaluate_graph_model(model, dataset, mask, criterion):
    model.eval()
    with torch.inference_mode():
        out = model(dataset.x, dataset.edge_index)
        score = criterion(out[mask].argmax(dim=1), dataset.y[mask])
    return score.item()

def evaluate_graph_training(model, dataset, criterion, training_results=None, plot_title="", savedir=None):
    if training_results:
        utils.plot_training_results(training_results, plot_title, savedir)
    train_score = evaluate_graph_model(
        model=model,
        dataset=dataset,
        mask=dataset.train_mask,
        criterion=criterion,
    )
    test_score = evaluate_graph_model(
        model=model,
        dataset=dataset,
        mask=dataset.test_mask,
        criterion=criterion,
    )
    print(f"Train accuracy: {train_score:.4f} | Test accuracy: {test_score:.4f}")

def evaluate_shadow_attack(attack_model, target_model, dataset, num_hops=0):
    attack_model.eval()
    target_model.eval()
    device = torch.device(next(attack_model.parameters()).device)
    with torch.inference_mode():
        features = query_attack_features(target_model, dataset, range(dataset.x.shape[0]), num_hops=num_hops)
        logits = attack_model(features)[:,1]
        labels = dataset.train_mask.long()
        return bc_evaluation(logits, labels, device)

def evaluate_confidence_attack(target_model, dataset, threshold, num_hops=0):
    target_model.eval()
    device = torch.device(next(target_model.parameters()).device)
    with torch.inference_mode():
        features = query_attack_features(target_model, dataset, range(dataset.x.shape[0]), num_hops=num_hops)
        confidences = F.softmax(features, dim=1).max(dim=1).values
        labels = dataset.train_mask.long()
        return bc_evaluation(confidences, labels, device, threshold=threshold)
