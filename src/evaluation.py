import datasetup
import utils

import numpy as np
import torch
from torch_geometric.utils import degree, k_hop_subgraph, subgraph
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

def k_hop_query(model, dataset, query_nodes, num_hops=0, inductive_split=False):
    '''
    Queries the model for each node in in query_nodes,
    using the local subgraph definded by the "num_hops"-hop neigborhood.
    When inductive_split flag is set, the local k-hop query is restricted to
    the nodes having the same training membership status as the center node.

    Output: Matrix of size "number of query nodes" times "number of classes",
            consisting of logits/predictions for each query node.
    '''
    model.eval()
    if not torch.is_tensor(query_nodes):
        query_nodes = torch.tensor(query_nodes, dtype=torch.int64)
    if query_nodes.shape == ():
        query_nodes.unsqueeze(dim=0)
    edge_mask = torch.ones(dataset.edge_index.shape[1], dtype=torch.bool).to(dataset.edge_index.device)
    if inductive_split:
        edge_mask = edge_mask & dataset.inductive_mask
    edge_index = dataset.edge_index[:, edge_mask]
    with torch.inference_mode():
        if num_hops == 0:
            empty_edge_index = torch.tensor([[],[]], dtype=torch.long).to(dataset.edge_index.device)
            preds = model(dataset.x[query_nodes], empty_edge_index)
        elif num_hops == model.num_layers:
            preds = model(dataset.x, edge_index)[query_nodes]
        else:
            preds = []
            for v in query_nodes:
                node_index, sub_edge_index, v_idx, _ = k_hop_subgraph(
                    node_idx=v.item(),
                    num_hops=num_hops,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=dataset.num_nodes,
                )
                pred = model(dataset.x[node_index], sub_edge_index)[v_idx].squeeze()
                preds.append(pred)
            preds = torch.stack(preds)
    assert preds.shape == (len(query_nodes), dataset.num_classes)
    return preds

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
