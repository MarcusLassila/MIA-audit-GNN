import utils

import torch
from torch_geometric.utils import k_hop_subgraph, subgraph, mask_to_index
from sklearn.metrics import roc_curve, roc_auc_score

def bc_evaluation(preds, labels):
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    auroc = roc_auc_score(y_true=labels, y_score=preds)
    fpr, tpr, _ = roc_curve(y_true=labels, y_score=preds)
    return {
        'auroc': auroc,
        'roc': (fpr, tpr),
    }

def k_hop_query(model, dataset, query_nodes, num_hops=0, use_ideal_neighborhood=True):
    '''
    Queries the model for each node in in query_nodes,
    using the local subgraph definded by the "num_hops"-hop neigborhood.
    When use_ideal_neighborhood flag is set, the local k-hop query is restricted to
    the nodes having the same training membership status as the center node.

    Output: Matrix of size "number of query nodes" times "number of classes",
            consisting of logits/predictions for each query node.
    '''
    model.eval()
    if not torch.is_tensor(query_nodes):
        query_nodes = torch.tensor(query_nodes, dtype=torch.int64)
    if num_hops == 0:
        with torch.inference_mode():
            predictions = model(dataset.x[query_nodes], torch.tensor([[],[]], dtype=torch.int64))
    else:
        # Can we speed this up?
        predictions = []
        for v in query_nodes:
            if use_ideal_neighborhood:
                edge_index = edge_index[:, dataset.inductive_mask]
            else:
                edge_index = dataset.edge_index
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
