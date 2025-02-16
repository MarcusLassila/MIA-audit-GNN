import models

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import degree, to_networkx, remove_isolated_nodes, index_to_mask
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve
from pathlib import Path
from time import perf_counter
from itertools import cycle, islice, product
import io
from contextlib import redirect_stdout, redirect_stderr

class Config:

    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def __str__(self):
        return '\n'.join(f'{k}: {v}'.replace('_', ' ') for k, v in self.__dict__.items())

def fresh_model(model_type, num_features, hidden_dims, num_classes, dropout=0.0):
    try:
        model = getattr(models, model_type)(
            in_dim=num_features,
            hidden_dims=hidden_dims,
            out_dim=num_classes,
            dropout=dropout,
        )
    except AttributeError:
        raise AttributeError(f'Unsupported model {model_type}. Supported models are MLP, GCN, SGC, GraphSAGE, GAT and GIN.')
    return model

def hinge_loss(pred, target):
    mask = torch.ones_like(pred, dtype=bool)
    mask[np.arange(target.shape[0]), target] = False
    return pred[~mask] - torch.max(pred[mask].reshape(target.shape[0], -1), dim=1).values

def measure_execution_time(callable):
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        ret = callable(*args, **kwargs)
        t1 = perf_counter()
        print(f"Callable '{callable.__name__}' executed in {t1 - t0:.3f} seconds.")
        return ret
    return wrapper

def average_degree(graph):
    return degree(graph.edge_index[0], num_nodes=graph.num_nodes, dtype=torch.float).mean().item()

def fraction_isolated_nodes(graph):
    _, _, mask = remove_isolated_nodes(graph.edge_index, num_nodes=graph.num_nodes)
    return (~mask).float().mean().item()

def execute_silently(callable, *args, **kwargs):
    null_stream = io.StringIO()
    with redirect_stdout(null_stream), redirect_stderr(null_stream):
        res = callable(*args, **kwargs)
    return res

def stat_repr(arr):
    ''' Return a string representation "mean (std)" of an array-like argument. '''
    arr = np.array(arr)
    return f'{arr.mean():.4f} ({arr.std():.4f})'

def graph_info(dataset):
    num_nodes = dataset.num_nodes
    num_edges = dataset.edge_index.shape[1]
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    class_counts = np.zeros(num_classes)
    for c in dataset.y:
        class_counts[c] += 1
    class_distr = class_counts / num_nodes
    avg_degree = average_degree(dataset)
    return (
        f'Dataset properties\n'
        f'#Nodes: {num_nodes}\n'
        f'#Edges: {num_edges}\n'
        f'#Features: {num_features}\n'
        f'#Classes: {num_classes}\n'
        f'#Class distribution: [{", ".join(f"{x:.4f}" for x in class_distr)}]\n'
        f'Average degree: {avg_degree:.4f}\n'
        f'Fraction isolated nodes: {fraction_isolated_nodes(dataset)}\n'
    )

def tpr_at_fixed_fpr(fpr, tpr, target_fpr, thresholds):
    idx = np.argmax(fpr >= target_fpr)
    if fpr[idx] > target_fpr + 1e-6:
        idx -= 1
    return tpr[idx], thresholds[idx]

def tpr_at_fixed_fpr_multi(soft_preds, truth, target_fpr):
    truth = truth.bool()
    num_pos = truth.sum().item()
    num_neg = truth.shape[0] - num_pos
    thresholds = []
    for soft_pred in soft_preds:
        fpr, tpr, threshold = roc_curve(y_true=truth, y_score=soft_pred)
        _, t = tpr_at_fixed_fpr(fpr, tpr, target_fpr, threshold)
        thresholds.append(threshold[np.where(threshold >= t)].tolist())
    best_tpr = 0.0
    best_thresholds = ()
    for ts in product(*thresholds):
        hard_pred = torch.zeros(soft_preds.shape[1], dtype=torch.bool)
        for soft_pred, threshold in zip(soft_preds, ts):
            hard_pred |= soft_pred >= threshold
        fpr = (hard_pred & ~truth).sum().item() / num_neg
        tpr = (hard_pred & truth).sum().item() / num_pos
        if fpr <= target_fpr and tpr > best_tpr:
            best_tpr = tpr
            best_thresholds = ts
    return best_tpr, best_thresholds

def min_max_normalization(*args):
    low = torch.inf
    high = -torch.inf
    for arg in args:
        low = min(low, arg.min())
        high = max(high, arg.max())
    return tuple((arg - low) / (high - low) for arg in args)

def partition_training_sets(num_nodes, num_models):
    '''Partition nodes such that each model is trained on half of the nodes. For e.g. LiRA and RMIA.'''
    train_masks = torch.zeros(size=(num_models, num_nodes), dtype=torch.bool)
    for i in range(num_nodes):
        models = np.random.choice(num_models, num_models // 2, replace=False)
        model_mask = index_to_mask(torch.from_numpy(models), size=num_models)
        for j in range(num_models):
            train_masks[j][i] = model_mask[j]
    return train_masks

########## Plotting helpers ##########

def plot_graph(graph):
    graph = to_networkx(graph)
    plt.figure(figsize=(10,8))
    nx.draw(graph, node_size=30, node_color='lightblue', with_labels=False, arrows=False, width=0.3, alpha=0.7)
    plt.title(graph.name)
    plt.show()

def plot_k_hop_subgraph(data, center_node, soft_preds_0, soft_preds_k, hard_preds_0, hard_preds_k, title):
    graph = to_networkx(data)
    node_colors = ['lightblue'] * graph.number_of_nodes()
    node_to_highlight = center_node
    node_colors[node_to_highlight] = 'red'
    labels = {
        i: f'{soft_preds_0[i].item():.1f}|{soft_preds_k[i].item():.1f}|{hard_preds_0[i].item()}-{hard_preds_k[i].item()}|{data.y[i].item()}'
        for i in range(data.num_nodes)
    }
    plt.figure(figsize=(10,8))
    plt.title(title)
    nx.draw(graph, labels=labels, node_size=500, node_color=node_colors, with_labels=True, arrows=False)
    plt.show()

def plot_training_results(res, name, savedir):
    epochs = np.array([*range(len(res['train_loss']))])
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, res['train_loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title('Training loss')
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, res['train_score'])
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title('Training accuracy')
    plt.grid(True)
    if 'valid_loss' in res:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, res['valid_loss'])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title('Validation loss')
        plt.grid(True)
        plt.subplot(2, 2, 4)
        plt.plot(epochs, res['valid_score'])
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.title('Validation accuracy')
        plt.grid(True)
    Path(savedir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{savedir}/training_results_{name}.png")
    plt.close()

def savefig_or_show(savepath=None):
    if savepath:
        savedir = '/'.join(savepath.split('/')[:-1])
        Path(savedir).mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

def plot_roc_loglog(fpr, tpr, title=None, savepath=None):
    plt.figure(figsize=(8, 8))
    plt.loglog(fpr, tpr)
    plt.xlim(1e-4, 1)
    plt.ylim(1e-4, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    savefig_or_show(savepath)

def plot_multi_roc_loglog(fprs, tprs, title=None, savepath=None):
    plt.figure(figsize=(8, 8))
    for fpr, tpr in islice(zip(fprs, tprs), 5):
        plt.loglog(fpr, tpr)
    plt.xlim(1e-4, 1)
    plt.ylim(1e-4, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    savefig_or_show(savepath)

def plot_roc_csv(filepath, savedir=None):
    df = pd.read_csv(filepath, sep=',')
    for s in df.keys():
        if s.endswith('fpr'):
            name = s[:-4]
            t = name + "_tpr"
            plot_roc_loglog(df[s], df[t], title=name, savepath=f'{savedir}/{name}.png')

def plot_histogram(x, bins, savepath=None):
    plt.figure(figsize=(10, 10))
    plt.hist(x=x, bins=bins)
    plt.grid(True)
    savefig_or_show(savepath)

def plot_histogram_and_fitted_gaussian(x, mean, std, bins=10, savepath=None):
    plt.figure(figsize=(8, 8))
    plt.hist(x=x, bins=bins, density=True)
    plt.grid(True)
    xmin, xmax = plt.xlim()
    xs = np.linspace(xmin, xmax, 100)
    ys = stats.norm.pdf(xs, loc=mean, scale=std)
    plt.plot(xs, ys, label='Gaussian fit')
    plt.title(f"Mean: {mean:.4f}, Std: {std:.4f}")
    savefig_or_show(savepath)

def plot_fitted_gaussians(means, stds, savepath=None):
    plt.figure(figsize=(8, 8))
    xs = np.linspace(-50, 50)
    for i, (mean, std) in enumerate(zip(means, stds)):
        ys = stats.norm.pdf(xs, loc=mean, scale=std+1e-6)
        plt.plot(xs, ys, label=f'{i}')
    plt.legend()
    plt.grid(True)
    savefig_or_show(savepath)

def plot_embedding_2D_scatter(embs, y, train_mask, savepath=None):
    length = train_mask.shape[0]
    trunc_length = 500
    if length > trunc_length:
        rand_index = torch.randint(low=0, high=length-1, size=(trunc_length,))
        embs = embs[rand_index]
        y = y[rand_index]
        train_mask = train_mask[rand_index]
    if embs.shape[1] > 2:
        x = TSNE(n_components=2).fit_transform(embs)
    else:
        x = embs
    plt.figure(figsize=(8, 8))
    colors = cycle(
        ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'yellow', 'cyan',
         'violet', 'grey', 'navy', 'deeppink', 'lawngreen', 'gold', 'teal']
    )
    for label in torch.unique(y):
        label_mask = y == label
        plt.scatter(x[train_mask & label_mask, 0], x[train_mask & label_mask, 1], c=next(colors), marker='o')
        plt.scatter(x[~train_mask & label_mask, 0], x[~train_mask & label_mask, 1], c=next(colors), marker='x')
    plt.grid(True)
    savefig_or_show(savepath)

def plot_embedding_hist(embs, mask, savepath=None):
    x = LinearDiscriminantAnalysis(n_components=1).fit_transform(X=embs, y=mask.long())
    bins = 50
    plt.figure(figsize=(8, 8))
    plt.hist(x=x[mask], bins=bins)
    plt.hist(x=x[~mask], bins=bins)
    plt.grid(True)
    savefig_or_show(savepath)

def plot_hinge_histogram(hinge, label_mask, train_mask, savepath=None):
    plt.figure(figsize=(8, 8))
    bins = min(50, 2 * len({x.item() for x in hinge}))
    plt.hist(hinge[train_mask & label_mask], bins=bins)
    plt.hist(hinge[~train_mask & label_mask], bins=bins)
    plt.grid(True)
    savefig_or_show(savepath)
