import engine
import models
import utils

import argparse

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--ds", default='Cora', type=str)
args = parser.parse_args()

class Config:
    batch_size = 32
    epochs     = 50
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    root       = "./data"
    hidden_dim = 256
    lr         = 1e-3

def get_dataset():
    if args.ds == "Cora":
        dataset = torch_geometric.datasets.CoraFull(Config.root + "/Cora")
    else:
        raise ValueError("Unsupported dataset!")
    return dataset

def run():
    dataset = get_dataset()
    model = models.GCN(
        dataset.num_features,
        Config.hidden_dim,
        dataset.num_classes
    ).to(Config.device)

    train_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    loss_fn = F.nll_loss
    res = engine.train(model, train_loader, None, loss_fn, optimizer, Config.epochs, Config.device)
    utils.plot_training_results(res)

if __name__ == '__main__':
    run()
