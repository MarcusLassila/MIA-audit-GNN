import engine
import models
import utils

import argparse

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--ds", default='Cora', type=str)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
args = parser.parse_args()

class Config:
    batch_size = args.batch_size
    epochs     = args.epochs
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    root       = "./data"
    hidden_dim = 256
    lr         = args.lr
    
    @staticmethod
    def print():
        print("\n########## Config ##########")
        print(f"Batch size: {Config.batch_size}")
        print(f"Epochs: {Config.epochs}")
        print(f"Learning rate: {Config.lr}")
        print(f"Device: {Config.device}")
        print(f"Hidden dimension: {Config.hidden_dim}")
        print("############################\n")

def get_dataset():
    if args.ds == "Cora":
        dataset = torch_geometric.datasets.Planetoid(root=Config.root, name="Cora", split="random", num_train_per_class=90)
    else:
        raise ValueError("Unsupported dataset!")
    return dataset

def main():
    dataset = get_dataset()
    model = models.GCN(
        dataset.num_features,
        Config.hidden_dim,
        dataset.num_classes
    ).to(Config.device)

    data_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    criterion = Accuracy(task='multiclass', num_classes=dataset.num_classes)
    loss_fn = F.nll_loss
    res = engine.train(model, data_loader, loss_fn, optimizer, criterion, Config.epochs, Config.device)
    utils.plot_training_results(res)

if __name__ == '__main__':
    Config.print()
    main()
