import models

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
from time import perf_counter

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

def plot_multi_roc_loglog(fprs, tprs, title=None, savepath=None):
    plt.figure(figsize=(8, 8))
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        plt.loglog(fpr, tpr, label=f'{i + 1}')
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
