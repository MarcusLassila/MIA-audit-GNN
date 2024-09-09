import utils

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
from sklearn.metrics import roc_curve, roc_auc_score
from math import comb

def inclusions(list_of_sets):
    res = [set() for _ in range(comb(len(list_of_sets)))]
    # TODO: Implement
    return res

def bc_evaluation(preds, truth, target_fpr):
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.cpu().numpy()
    auroc = roc_auc_score(y_true=truth, y_score=preds)
    fpr, tpr, thresholds = roc_curve(y_true=truth, y_score=preds)
    tpr_fixed_fpr, threshold = utils.tpr_at_fixed_fpr(fpr, tpr, target_fpr, thresholds)
    hard_preds = (preds >= threshold).astype(np.int64)
    true_positives = (hard_preds & truth).nonzero()[0]
    return {
        'auroc': auroc,
        'roc': (fpr, tpr),
        'tpr_fixed_fpr': tpr_fixed_fpr,
        'TP_fixed_fpr': true_positives,
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
    with torch.inference_mode():
        if num_hops == 0:
            empty_edge_index = torch.tensor([[],[]], dtype=torch.int64).to(dataset.edge_index.device)
            predictions = model(dataset.x[query_nodes], empty_edge_index)
        elif num_hops == model.num_propagations:
            edge_index = dataset.edge_index[:, dataset.inductive_mask] if inductive_split else dataset.edge_index[:, dataset.random_edge_mask]
            predictions = model(dataset.x[query_nodes], edge_index)
        else:
            edge_index = dataset.edge_index[:, dataset.inductive_mask] if inductive_split else dataset.edge_index[:, dataset.random_edge_mask]
            predictions = []
            for v in query_nodes:
                node_index, edge_index, v_idx, _ = k_hop_subgraph(
                    node_idx=v.item(),
                    num_hops=num_hops,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=dataset.x.shape[0],
                )
                pred = model(dataset.x[node_index], edge_index)[v_idx].squeeze()
                predictions.append(pred)
            predictions = torch.stack(predictions)
    assert predictions.shape == torch.Size([len(query_nodes), dataset.num_classes])
    return predictions

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
