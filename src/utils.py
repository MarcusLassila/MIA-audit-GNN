import models

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from time import perf_counter

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

def fresh_model(model_type, num_features, hidden_dim, num_classes, dropout=0.0):
    try:
        model = getattr(models, model_type)(
            in_dim=num_features,
            hidden_dim=hidden_dim,
            out_dim=num_classes,
            dropout=dropout,
        )
    except AttributeError:
        raise AttributeError(f'Unsupported model {model_type}. Supported models are GCN, SGC, GraphSAGE, GAT and GIN.')
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

def plot_roc_loglog(fpr, tpr, title=None, savepath=None):
    plt.figure(figsize=(8, 8))
    plt.loglog(fpr, tpr)
    plt.xlim(1e-4, 1)
    plt.ylim(1e-4, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    if savepath:
        savedir = '/'.join(savepath.split('/')[:-1])
        Path(savedir).mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

def plot_multi_roc_loglog(fprs, tprs, test_accs, title=None, savepath=None):
    plt.figure(figsize=(8, 8))
    for fpr, tpr, acc in zip(fprs, tprs, test_accs):
        plt.loglog(fpr, tpr, label=f'acc: {acc:.4f}')
    plt.xlim(1e-4, 1)
    plt.ylim(1e-4, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.title(title)
    if savepath:
        savedir = '/'.join(savepath.split('/')[:-1])
        Path(savedir).mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

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
    if savepath:
        savedir = '/'.join(savepath.split('/')[:-1])
        Path(savedir).mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()
