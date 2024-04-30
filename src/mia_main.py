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
from torchmetrics import Accuracy, AUROC, F1Score
from sklearn.model_selection import train_test_split

torch.random.manual_seed(666)

parser = argparse.ArgumentParser()
parser.add_argument("--ds", default='cora', type=str)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--model", default="GCN", type=str)
args = parser.parse_args()

class Config:
    batch_size = args.batch_size
    epochs     = args.epochs
    dataset    = args.ds
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    root       = "./data"
    hidden_dim = 256
    lr         = args.lr
    model      = args.model
    
    @staticmethod
    def print():
        print("\n########## Config ##########")
        print(f"Model: {args.model}")
        print(f"Dataset: {Config.dataset}")
        print(f"Batch size: {Config.batch_size}")
        print(f"Epochs: {Config.epochs}")
        print(f"Learning rate: {Config.lr}")
        print(f"Device: {Config.device}")
        print(f"Hidden dimension: {Config.hidden_dim}")
        print("############################\n")

def get_dataset():
    if Config.dataset == "cora":
        dataset = torch_geometric.datasets.Planetoid(root=Config.root, name="Cora", split="random", num_train_per_class=90)
    elif Config.dataset == "citeseer":
        dataset = torch_geometric.datasets.Planetoid(root=Config.root, name="CiteSeer", split="random", num_train_per_class=100)
    elif Config.dataset == "pubmed":
        dataset = torch_geometric.datasets.Planetoid(root=Config.root, name="PubMed", split="random", num_train_per_class=1500)
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
    else:
        raise ValueError("Unsupported model!")
    return model

def get_criterion(dataset):
    return Accuracy(task='multiclass', num_classes=dataset.num_classes)

def create_attack_dataset(shadow_dataset, shadow_model):
    features = shadow_model(shadow_dataset.x, shadow_dataset.edge_index)
    labels = shadow_dataset.train_mask.long()
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=50, stratify=labels, random_state=777)
    train_dataset = utils.AttackDataset(train_X, train_y)
    test_dataset = utils.AttackDataset(test_X, test_y)
    return train_dataset, test_dataset

def train_graph_model(dataset, model, name):
    ''' Train target or shadow model. '''
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    criterion = get_criterion(dataset).to(Config.device)
    loss_fn = F.nll_loss
    res = trainer.train_gnn(model, dataset, loss_fn, optimizer, criterion, Config.epochs, Config.device)
    utils.plot_training_results(res, name)
    Path('models').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"models/{name}_{model.__class__.__name__}_{dataset.name}.pth")
    test_score = infer.test(model, dataset, criterion)
    print(f"{name} test score: {test_score:.4f}")

def train_attack(model, train_dataset, test_dataset):
    ''' Train attack model. '''
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = F1Score(task="multiclass", num_classes=2)
    loss_fn = nn.CrossEntropyLoss()
    res = trainer.train_mlp(model, train_loader, test_loader, loss_fn, optimizer, criterion, epochs=100, device=Config.device)
    utils.plot_training_results(res, name='Attack')

def main():
    target_dataset = get_dataset()
    shadow_dataset = get_dataset()
    target_model = get_model(target_dataset)
    shadow_model = get_model(shadow_dataset)
    train_graph_model(target_dataset, target_model, 'Target')
    train_graph_model(shadow_dataset, shadow_model, 'Shadow')

    train_dataset, test_dataset = create_attack_dataset(shadow_dataset, shadow_model)
    attack_model = models.MLP(in_dim=shadow_dataset.num_classes)
    train_attack(attack_model, train_dataset, test_dataset)

    features = target_model(target_dataset.x, target_dataset.edge_index)
    ground_truth = target_dataset.train_mask.long()
    attack_dataset = utils.AttackDataset(features, ground_truth)
    infer.evaluate_attack_model(attack_model, attack_dataset)

if __name__ == '__main__':
    Config.print()
    main()
