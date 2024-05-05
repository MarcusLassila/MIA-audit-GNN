import torch_geometric.data
import torch_geometric.datasets
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
from sklearn.model_selection import train_test_split
from statistics import mean, stdev

torch.random.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='cora', type=str)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--epochs-attack", default=100, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--model", default="GCN", type=str)
parser.add_argument("--savedir", default="plots", type=str)
parser.add_argument("--experiments", default=1, type=int)
parser.add_argument("--id", default=None, type=str)
args = parser.parse_args()

class Config:
    batch_size    = args.batch_size
    epochs        = args.epochs
    epochs_attack = args.epochs_attack
    dataset       = args.dataset
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    root          = "./data"
    hidden_dim    = 256
    lr            = args.lr
    model         = args.model
    savedir       = args.savedir
    experiments   = args.experiments
    identifier    = args.id or args.model + "_" + args.dataset
    
    @staticmethod
    def print():
        print("\n########## Config ##########")
        print(f"Number of experiments: {Config.experiments}")
        print(f"Model: {args.model}")
        print(f"Dataset: {Config.dataset}")
        print(f"Batch size: {Config.batch_size}")
        print(f"Epochs target/shadow: {Config.epochs}")
        print(f"Epochs attack")
        print(f"Learning rate (target/shadow): {Config.lr}")
        print(f"Device: {Config.device}")
        print("############################\n")

def get_dataset():
    if Config.dataset == "cora":
        dataset = torch_geometric.datasets.Planetoid(root=Config.root, name="Cora", split="random", num_train_per_class=90)
    elif Config.dataset == "citeseer":
        dataset = torch_geometric.datasets.Planetoid(root=Config.root, name="CiteSeer", split="random", num_train_per_class=100)
    elif Config.dataset == "pubmed":
        dataset = torch_geometric.datasets.Planetoid(root=Config.root, name="PubMed", split="random", num_train_per_class=1500)
    elif Config.dataset == "flickr":
        dataset = torch_geometric.datasets.Flickr(root=Config.root)
        dataset.name = 'Flickr'
    else:
        raise ValueError("Unsupported dataset!")
    return dataset

def get_model(dataset):
    if Config.model == "GCN":
        model = models.GCN(
            dataset.num_features,
            Config.hidden_dim,
            dataset.num_classes,
        )
    elif Config.model == 'GAT':
        model = models.GAT(
            dataset.num_features,
            Config.hidden_dim,
            dataset.num_classes,
        )
    else:
        raise ValueError("Unsupported model!")
    return model

def get_criterion(dataset):
    return Accuracy(task='multiclass', num_classes=dataset.num_classes).to(Config.device)

def create_attack_dataset(shadow_dataset, shadow_model):
    features = shadow_model(shadow_dataset.x, shadow_dataset.edge_index).cpu()
    labels = shadow_dataset.train_mask.long().cpu()
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=50, stratify=labels) # test_size=0.2
    train_dataset = utils.AttackDataset(train_X, train_y)
    test_dataset = utils.AttackDataset(test_X, test_y)
    return train_dataset, test_dataset

def train_graph_model(dataset, model, name):
    ''' Train target or shadow model. '''
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    criterion = get_criterion(dataset)
    loss_fn = F.nll_loss
    res = trainer.train_gnn(model, dataset, loss_fn, optimizer, criterion, Config.epochs, Config.device)
    utils.plot_training_results(res, name, Config.savedir)
    Path('models').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"models/{name}_{model.__class__.__name__}_{dataset.name}.pth")
    test_score = infer.evaluate_graph_model(model, dataset, dataset.test_mask, criterion)
    print(f"{name} test score: {test_score:.4f}")

def train_attack(model, train_dataset, valid_dataset):
    ''' Train attack model. '''
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = Accuracy(task="multiclass", num_classes=2).to(Config.device)
    loss_fn = nn.CrossEntropyLoss()
    res = trainer.train_mlp(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        criterion=criterion,
        epochs=Config.epochs_attack,
        device=Config.device
    )
    utils.plot_training_results(res, name='Attack', savedir=Config.savedir)

def run_experiment(seed):
    torch.manual_seed(seed)
    target_dataset = get_dataset()
    shadow_dataset = get_dataset()
    target_model = get_model(target_dataset)
    shadow_model = get_model(shadow_dataset)
    train_graph_model(target_dataset, target_model, 'Target')
    train_graph_model(shadow_dataset, shadow_model, 'Shadow')
    
    target_scores = {
        'train_score': infer.evaluate_graph_model(target_model, target_dataset, target_dataset.train_mask, get_criterion(target_dataset)),
        'test_score': infer.evaluate_graph_model(target_model, target_dataset, target_dataset.test_mask, get_criterion(target_dataset))
    }

    train_dataset, valid_dataset = create_attack_dataset(shadow_dataset, shadow_model)
    attack_model = models.MLP(in_dim=shadow_dataset.num_classes)
    train_attack(attack_model, train_dataset, valid_dataset)

    features = target_model(target_dataset.x, target_dataset.edge_index)
    ground_truth = target_dataset.train_mask.long()
    attack_dataset = utils.AttackDataset(features, ground_truth)
    eval_metrics = infer.evaluate_attack_model(attack_model, attack_dataset, Config.device)
    return dict(target_scores, **eval_metrics)

def main():
    aurocs, f1s, precisions, recalls = [], [], [], []
    best_auroc, best_roc = 0, ()
    for i in range(Config.experiments):
        print(f'Running experiment {i + 1}.')
        metrics = run_experiment(i)
        if best_auroc < metrics['auroc']:
            best_auroc = metrics['auroc']
            best_roc = metrics['roc']
        aurocs.append(metrics['auroc'])
        f1s.append(metrics['f1_score'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
    if Config.experiments > 1:
        auroc = (mean(aurocs), stdev(aurocs))
        f1_score = (mean(f1s), stdev(f1s))
        precision = (mean(precisions), stdev(precisions))
        recall = (mean(recalls), stdev(recalls))
    else:
        auroc = (aurocs[0], 0)
        f1_score = (f1s[0], 0)
        precision = (precisions[0], 0)
        recall = (recalls[0], 0)
    with open("MIA_output.txt", "a") as f:
        f.write(Config.identifier + '\n')
        f.write(str({
            'train_acc': metrics['train_score'],
            'test_acc': metrics['test_score'],
            'auroc': auroc,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
        }))
        f.write('\n')
    print()
    print(f"Target train score: {metrics['train_score']:.4f}")
    print(f"Target test score: {metrics['test_score']:.4f}")
    print(f"Auroc: {auroc[0]:.4f} +- {auroc[1]:.4f}")
    print(f"F1 score: {f1_score[0]:.4f} +- {f1_score[1]:.4f}")
    print(f"Precision: {precision[0]:.4f} +- {precision[1]:.4f}")
    print(f"Recall: {recall[0]:.4f} +- {recall[1]:.4f}")
    utils.plot_roc_loglog(*best_roc, name=f"{Config.model}_{Config.dataset}", savedir=Config.savedir)

if __name__ == '__main__':
    Config.print()
    main()
