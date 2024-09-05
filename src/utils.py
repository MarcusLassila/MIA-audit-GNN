import models

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors
import torch
import torch_geometric.nn as gnn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pathlib import Path
from time import perf_counter
from itertools import cycle, islice

class Config:

    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def __str__(self):
        return '\n'.join(f'{k}: {v}'.replace('_', ' ') for k, v in self.__dict__.items())

class GraphInfo:

    def __init__(self, dataset):
        self.name = dataset.name
        self.num_nodes = dataset.x.shape[0]
        self.num_edges = dataset.edge_index.shape[1]
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.class_counts = np.zeros(self.num_classes)
        for c in dataset.y:
            self.class_counts[c] += 1
        self.class_distr = self.class_counts / self.num_nodes

    def __str__(self):
        s = (
            f'Dataset: {self.name}\n'
            f'#Nodes: {self.num_nodes}\n'
            f'#Edges: {self.num_edges}\n'
            f'#Features: {self.num_features}\n'
            f'#Classes: {self.num_classes}\n'
            f'#Class distribution: [{", ".join(f"{x:.4f}" for x in self.class_distr)}]\n'
        )
        return s

def fresh_model(model_type, num_features, hidden_dims, num_classes, dropout=0.0):
    if model_type == 'MLP':
        return gnn.MLP(channel_list=[num_features, *hidden_dims, num_classes], dropout=dropout)
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

def tpr_at_fixed_fpr(fpr, tpr, target_fpr):
    idx = np.argmax(fpr >= target_fpr)
    if fpr[idx] != target_fpr:
        x0, x1 = fpr[idx - 1], fpr[idx]
        y0, y1 = tpr[idx - 1], tpr[idx]
        slope = (y1 - y0) / (x1 - x0)
        return slope * (target_fpr - x0) + y0
    else:
        return tpr[idx]

def plot_training_results(res, name, savedir):
    epochs = np.array([*range(len(res['train_loss']))])
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, res['train_loss'], label='train loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, res['train_score'], label='train score')
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, res['valid_loss'], label='valid loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, res['valid_score'], label='valid score')
    plt.xlabel("Epochs")
    plt.ylabel("Score")
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

def plot_multi_roc_loglog(fprs_list, tprs_list, query_hops, title=None, savepath=None):
    plt.figure(figsize=(8, 8))
    for fprs, tprs, num_hops in zip(fprs_list, tprs_list, query_hops):
        for fpr, tpr in islice(zip(fprs, tprs), 5):
            plt.loglog(fpr, tpr, label=f'{num_hops}-hop')
    plt.xlim(1e-4, 1)
    plt.ylim(1e-4, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.title(title)
    savefig_or_show(savepath)

def plot_roc_csv(filepath, savedir=None):
    df = pd.read_csv(filepath, sep=',')
    for s in df.keys():
        if s.endswith('fpr'):
            name = s[:-4]
            t = name + "_tpr"
            plot_roc_loglog(df[s], df[t], title=name, savepath=f'{savedir}/{name}.png')

def plot_histogram_and_fitted_gaussian(x, mean, std, bins=10, savepath=None):
    plt.figure(figsize=(8, 8))
    plt.hist(x=x, bins=bins, density=True)
    plt.grid(True)
    xmin, xmax = plt.xlim()
    xs = np.linspace(xmin, xmax)
    ys = stats.norm.pdf(xs, loc=mean, scale=std)
    plt.plot(xs, ys, label='Gaussian fit')
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
