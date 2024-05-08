import data
import infer
import models
import utils
import trainer

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric
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
    if CONFIG.dataset == "cora":
        dataset = data.Cora(root=CONFIG.datadir, split=CONFIG.split)
    elif CONFIG.dataset == "citeseer":
        dataset = torch_geometric.datasets.Planetoid(root=CONFIG.datadir, name="CiteSeer", split="random", num_train_per_class=100)
    elif CONFIG.dataset == "pubmed":
        dataset = torch_geometric.datasets.Planetoid(root=CONFIG.datadir, name="PubMed", split="random", num_train_per_class=1500)
    elif CONFIG.dataset == "flickr":
        dataset = torch_geometric.datasets.Flickr(root=CONFIG.datadir)
        dataset.name = 'Flickr'
    else:
        raise ValueError("Unsupported dataset!")
    return dataset

def get_model(dataset):
    try:
        model = getattr(models, CONFIG.model)(
            dataset.num_features,
            CONFIG.hidden_dim_target,
            dataset.num_classes,
            dropout=CONFIG.dropout,
        )
    except AttributeError:
        err = AttributeError(f'Unsupported model {CONFIG.model}')
        raise err
    return model

def get_criterion(dataset):
    return Accuracy(task='multiclass', num_classes=dataset.num_classes).to(CONFIG.device)

def train_graph_model(dataset, model, name):
    ''' Train target or shadow model. '''
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.lr)
    criterion = get_criterion(dataset)
    loss_fn = F.nll_loss
    res = trainer.train_gnn(model, dataset, loss_fn, optimizer, criterion, CONFIG.epochs_target, CONFIG.device)
    utils.plot_training_results(res, name, CONFIG.savedir)
    Path('models').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"models/{name}_{model.__class__.__name__}_{dataset.name}.pth")
    test_score = infer.evaluate_graph_model(model, dataset, dataset.test_mask, criterion)
    print(f"{name} test score: {test_score:.4f}")

def train_attack(model, train_dataset, valid_dataset):
    ''' Train attack model. '''
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = Accuracy(task="multiclass", num_classes=2).to(CONFIG.device)
    loss_fn = nn.CrossEntropyLoss()
    res = trainer.train_mlp(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        criterion=criterion,
        epochs=CONFIG.epochs_attack,
        device=CONFIG.device
    )
    utils.plot_training_results(res, name='Attack', savedir=CONFIG.savedir)

def run_experiment(seed):
    torch.manual_seed(seed)
    if CONFIG.dataset == 'cora':
        target_dataset, shadow_dataset = get_dataset().get_split()
    else:
        target_dataset = get_dataset()
        shadow_dataset = get_dataset()
    target_model = get_model(target_dataset)
    shadow_model = get_model(shadow_dataset)
    train_graph_model(target_dataset, target_model, 'Target')
    train_graph_model(shadow_dataset, shadow_model, 'Shadow')
    
    criterion = get_criterion(target_dataset)
    target_scores = {
        'train_score': infer.evaluate_graph_model(target_model, target_dataset, target_dataset.train_mask, criterion),
        'test_score': infer.evaluate_graph_model(target_model, target_dataset, target_dataset.test_mask, criterion)
    }

    train_dataset, valid_dataset = data.create_attack_dataset(shadow_dataset, shadow_model)
    attack_model = models.MLP(in_dim=shadow_dataset.num_classes, hidden_dims=CONFIG.hidden_dim_attack)
    train_attack(attack_model, train_dataset, valid_dataset)

    eval_metrics = infer.evaluate_attack_model(attack_model, target_model, target_dataset, num_hops=CONFIG.query_hops)
    return dict(target_scores, **eval_metrics)

def main(config):
    global CONFIG
    CONFIG = Objectify(config)
    CONFIG.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_scores, test_scores = [], []
    aurocs, f1s, precisions, recalls = [], [], [], []
    best_auroc, best_roc = 0, ()
    for i in range(CONFIG.experiments):
        print(f'Running experiment {i + 1}/{CONFIG.experiments}.')
        metrics = run_experiment(i)
        if best_auroc < metrics['auroc']:
            best_auroc = metrics['auroc']
            best_roc = metrics['roc']
        train_scores.append(metrics['train_score'])
        test_scores.append(metrics['test_score'])
        aurocs.append(metrics['auroc'])
        f1s.append(metrics['f1_score'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
    if CONFIG.experiments > 1:
        with open(CONFIG.outputfile, "a") as file:
            file.write(CONFIG.name + ':\n')
            file.write(f'  train_score_mean: {mean(train_scores)}\n')
            file.write(f'  train_score_stdev: {stdev(train_scores)}\n')
            file.write(f'  test_score_mean: {mean(test_scores)}\n')
            file.write(f'  test_score_std: {stdev(test_scores)}\n')
            file.write(f'  auroc_mean: {mean(aurocs)}\n')
            file.write(f'  auroc_stdev: {stdev(aurocs)}\n')
            file.write(f'  f1_score_mean: {mean(f1s)}\n')
            file.write(f'  f1_score_stdev: {stdev(f1s)}\n')
            file.write(f'  precision_mean: {mean(precisions)}\n')
            file.write(f'  precision_stdev: {stdev(precisions)}\n')
            file.write(f'  recall_mean: {mean(recalls)}\n')
            file.write(f'  recall_stdev: {stdev(recalls)}\n')
    else:
        with open(CONFIG.outputfile, "a") as file:
            file.write(CONFIG.name + ':\n')
            file.write(f'  train_score: {train_scores[0]}\n')
            file.write(f'  test_score: {test_scores[0]}\n')
            file.write(f'  auroc: {aurocs[0]}\n')
            file.write(f'  f1_score: {f1s[0]}\n')
            file.write(f'  precision: {precisions[0]}\n')
            file.write(f'  recall: {recalls[0]}\n')
    utils.plot_roc_loglog(*best_roc, name=CONFIG.name, savedir=CONFIG.savedir) # Plot the ROC curve for sample with highest AUROC.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cora', type=str)
    parser.add_argument("--split", default="sampled", type=str)
    parser.add_argument("--model", default="GCN", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs-target", default=50, type=int)
    parser.add_argument("--epochs-attack", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--hidden-dim-target", default=256, type=int)
    parser.add_argument("--hidden-dim-attack", default=[100, 50], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--query-hops", default=0, type=int)
    parser.add_argument("--experiments", default=1, type=int)
    parser.add_argument("--name", default="unnamed", type=str)
    parser.add_argument("--datadir", default="./data", type=str)
    parser.add_argument("--savedir", default="./plots", type=str)
    parser.add_argument("--outputfile", default="output.yaml", type=str)
    args = parser.parse_args()
    open(args.outputfile, "w").close() # Clear output file.
    config = vars(args)
    print('Running MIA experiment.')
    print(Objectify(config))
    main(config)
