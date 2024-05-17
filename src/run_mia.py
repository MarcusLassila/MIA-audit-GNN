import attacks
import datasetup
import evaluation
import models
import utils
import trainer

import argparse
from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.datasets
from torchmetrics import Accuracy, Precision, Recall, F1Score
from statistics import mean, stdev

torch.random.manual_seed(0)

CONFIG = None

class Objectify:

    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.__dict__.update(dictionary)
    
    def __str__(self):
        return '\n'.join(f'{k}: {v}'.replace('_', ' ') for k, v in self.dictionary.items())

def get_dataset():
    root = CONFIG.datadir
    if CONFIG.dataset == "cora":
        dataset = torch_geometric.datasets.Planetoid(root=root, name="Cora")
    elif CONFIG.dataset == "corafull":
        dataset = torch_geometric.datasets.CoraFull(root=root)
        dataset.name == "CoraFull"
    elif CONFIG.dataset == "citeseer":
        dataset = torch_geometric.datasets.Planetoid(root=root, name="CiteSeer")
    elif CONFIG.dataset == "pubmed":
        dataset = torch_geometric.datasets.Planetoid(root=root, name="PubMed")
    elif CONFIG.dataset == "flickr":
        dataset = torch_geometric.datasets.Flickr(root=root)
        dataset.name = "Flickr"
    else:
        raise ValueError("Unsupported dataset!")
    return dataset

def get_model(in_dim, out_dim):
    try:
        model = getattr(models, CONFIG.model)(
            in_dim=in_dim,
            hidden_dim=CONFIG.hidden_dim_target,
            out_dim=out_dim,
            dropout=CONFIG.dropout,
        )
    except AttributeError:
        raise AttributeError(f'Unsupported model {CONFIG.model}. Supported models are GCN, SGC, GraphSAGE, GAT and GIN.')
    return model

def main(config):
    global CONFIG
    CONFIG = Objectify(config)
    CONFIG.dataset = CONFIG.dataset.lower()
    CONFIG.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_scores, test_scores = [], []
    aurocs, f1s, precisions, recalls = [], [], [], []
    best_auroc = 0
    dataset = get_dataset()
    for i in range(CONFIG.experiments):
        print(f'Running experiment {i + 1}/{CONFIG.experiments}.')
        
        if CONFIG.attack == "basic-shadow":
            target_model = get_model(dataset.num_features, dataset.num_classes)
            shadow_model = get_model(dataset.num_features, dataset.num_classes)
            metrics = attacks.BasicShadowAttack(dataset=dataset, target_model=target_model, shadow_model=shadow_model, config=CONFIG).run_attack()
        elif CONFIG.attack == "confidence":
            target_model = get_model(dataset.num_features, dataset.num_classes)
            target_dataset = datasetup.sample_subgraph(dataset, num_nodes=dataset.x.shape[0]//2)
            print(target_dataset)
            metrics = attacks.ConfidenceAttack(dataset=target_dataset, target_model=target_model, config=CONFIG).run_attack()

        if best_auroc < metrics['auroc']:
            best_auroc = metrics['auroc']
            fpr, tpr = metrics['roc']
        train_scores.append(metrics['train_score'])
        test_scores.append(metrics['test_score'])
        aurocs.append(metrics['auroc'])
        f1s.append(metrics['f1_score'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
    if CONFIG.experiments > 1:
        stats = {
            'train_acc_mean': [mean(train_scores)],
            'train_acc_stdev': [stdev(train_scores)],
            'test_acc_mean': [mean(test_scores)],
            'test_acc_stdev': [stdev(test_scores)],
            'auroc_mean': [mean(aurocs)],
            'auroc_stdev': [stdev(aurocs)],
            'f1_score_mean': [mean(f1s)],
            'f1_score_stdev': [stdev(f1s)],
            'precision_mean': [mean(precisions)],
            'precision_stdev': [stdev(precisions)],
            'recall_mean': [mean(recalls)],
            'recall_stdev': [stdev(recalls)],
        }
    else:
        stats = {
            'train_acc': train_scores,
            'test_acc': test_scores,
            'auroc': aurocs,
            'f1_score': f1s,
            'precision': precisions,
            'recall': recalls,
        }
    stat_df = pd.DataFrame(stats, index=[CONFIG.name])
    roc_df = pd.DataFrame({f'{CONFIG.name}_fpr': fpr, f'{CONFIG.name}_tpr': tpr})
    if CONFIG.plot_roc:
        savepath = f'{CONFIG.savedir}/{CONFIG.name}_roc_loglog.png'
        utils.plot_roc_loglog(fpr, tpr, savepath=savepath) # Plot the ROC curve for sample with highest AUROC.
    return stat_df, roc_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="basic-shadow", type=str)
    parser.add_argument("--dataset", default="cora", type=str)
    parser.add_argument("--split", default="sampled", type=str)
    parser.add_argument("--model", default="GCN", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs-target", default=100, type=int)
    parser.add_argument("--epochs-attack", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction)
    parser.add_argument("--hidden-dim-target", default=256, type=int)
    parser.add_argument("--hidden-dim-attack", default=[128, 64], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--query-hops", default=0, type=int)
    parser.add_argument("--experiments", default=1, type=int)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--confidence-threshold", default=0.5, type=float)
    parser.add_argument("--name", default="unnamed", type=str)
    parser.add_argument("--datadir", default="./data", type=str)
    parser.add_argument("--savedir", default="./results", type=str)
    parser.add_argument("--plot-roc", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    config = vars(args)
    print('Running MIA experiment.')
    print(Objectify(config))
    print()
    stat_df, roc_df = main(config)
    print('Attack statistics:')
    print(stat_df)
    roc_df.to_csv(f'{args.savedir}/roc_{args.name}.csv', index=False)
