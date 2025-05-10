import datasetup
import utils

import numpy as np
import torch
from torch_geometric.utils import degree
from sklearn.metrics import roc_curve, roc_auc_score
from itertools import combinations

def inclusions(list_of_sets):
    n = len(list_of_sets)
    res = []
    for r in range(1, n + 1):
        for idx in combinations(range(n), r):
            incl = set()
            excl = set()
            for i in range(n):
                if i in idx:
                    incl.update(list_of_sets[i])
                else:
                    excl.update(list_of_sets[i])
            res.append(len(incl - excl))
    return res

def evaluate_binary_classification(preds, truth, target_fpr, target_node_index, graph, top_k=20):
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.cpu().numpy()
    auroc = roc_auc_score(y_true=truth, y_score=preds)
    fpr, tpr, thresholds = roc_curve(y_true=truth, y_score=preds)
    tpr_fixed_fpr = []
    threshold_fixed_fpr = []
    for t_fpr in target_fpr:
        t_tpr, threshold = utils.tpr_at_fixed_fpr(fpr, tpr, t_fpr, thresholds)
        tpr_fixed_fpr.append(t_tpr)
        threshold_fixed_fpr.append(threshold)
    _, top_k_index = torch.topk(torch.from_numpy(preds), top_k)
    top_k_nodes = target_node_index[top_k_index]
    degree_top_k = degree(graph.edge_index[0, graph.inductive_mask], graph.num_nodes, dtype=torch.long)[top_k_nodes].cpu().numpy()
    return {
        'AUC': auroc,
        'ROC': (fpr, tpr),
        'TPR@FPR': tpr_fixed_fpr,
        'threshold@FPR': threshold_fixed_fpr,
        'degree_top_k': degree_top_k,
    }

def evaluate_graph_model(model, dataset, mask, criterion, inductive_inference):
    model.eval()
    with torch.inference_mode():
        if inductive_inference:
            out = model(dataset.x, dataset.edge_index[:, dataset.inductive_mask])
        else:
            out = model(dataset.x, dataset.edge_index)
        score = criterion(out[mask].argmax(dim=1), dataset.y[mask])
    return score.item()

def evaluate_graph_training(model, dataset, criterion, inductive_inference=True, training_results=None, plot_name="", savedir=None):
    if training_results:
        utils.plot_training_results(training_results, plot_name, savedir)
    train_score = evaluate_graph_model(
        model=model,
        dataset=dataset,
        mask=dataset.train_mask,
        criterion=criterion,
        inductive_inference=inductive_inference,
    )
    test_score = evaluate_graph_model(
        model=model,
        dataset=dataset,
        mask=dataset.test_mask,
        criterion=criterion,
        inductive_inference=inductive_inference,
    )
    print(f"Train accuracy: {train_score:.4f} | Test accuracy: {test_score:.4f}")
