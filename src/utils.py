import models

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

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

def plot_roc_loglog(fpr, tpr, name=None, savepath=None):
    plt.figure(figsize=(8, 8))
    plt.loglog(fpr, tpr)
    plt.xlim(1e-4, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(name)
    if savepath:
        savedir = '/'.join(savepath.split('/')[:-1])
        Path(savedir).mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

def plot_roc_csv(filepath, savepath=None):
    df = pd.read_csv(filepath, sep=',')
    for fpr in filter(lambda s: s.endswith('fpr'), df.keys()):
        name = fpr[:-4]
        tpr = name + "_tpr"
        plot_roc_loglog(df[fpr], df[tpr], name=name, savepath=savepath)

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
