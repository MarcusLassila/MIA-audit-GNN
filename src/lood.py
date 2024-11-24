import datasetup
import trainer
import utils

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import subgraph
from torchmetrics import Accuracy
from tqdm.auto import tqdm

def gaussian_kl_divergence(mean_0, std_0, mean_1, std_1):
    var_0 = std_0 ** 2
    var_1 = std_1 ** 2
    return torch.log(std_1 / std_0) - 0.5 + (var_0 + (mean_0 - mean_1) ** 2) / (2 * var_1)

class LOOD:
    
    def __init__(self, config):
        self.config = config

    def train_model(self, dataset, hidden_dims, dropout, disable_tqdm=True):
        config = self.config
        model = utils.fresh_model(
            model_type=self.config.model,
            num_features=dataset.num_features,
            hidden_dims=hidden_dims,
            num_classes=dataset.num_classes,
            dropout=dropout,
        )
        train_config = trainer.TrainConfig(
            criterion=Accuracy(task="multiclass", num_classes=dataset.num_classes).to(self.config.device),
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=F.cross_entropy,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        _ = trainer.train_gnn(
            model=model,
            dataset=dataset,
            config=train_config,
            disable_tqdm=disable_tqdm,
            inductive_split=config.inductive_split,
        )
        model.eval()
        return model

    def quantify_query_distributions(self, dataset):
        config = self.config

        preds = []
        for _ in tqdm(range(5000), desc="Training shadow models"):
            model = self.train_model(dataset, hidden_dims=[64], dropout=0.0)
            with torch.no_grad():
                preds.append(model(dataset.x, dataset.edge_index)[0, dataset.y[0]])
        preds = torch.tensor(preds)
        mean = preds.mean()
        std = preds.std()
        print('Retrain')
        print(f'mean {mean:.4f}, std: {std:.4f}')
        utils.plot_histogram_and_fitted_gaussian(preds, mean, std, bins=50, savepath=f'{config.savedir}/retrain.png')

        preds = []
        for _ in tqdm(range(50), desc="Training shadow models"):
            model = self.train_model(dataset, hidden_dims=[128], dropout=0.5)
            model.dropout_during_inference = True
            with torch.no_grad():
                for _ in range(1000):
                    preds.append(model(dataset.x, dataset.edge_index)[0, dataset.y[0]])
        preds = torch.tensor(preds)
        mean = preds.mean()
        std = preds.std()
        print('Hybrid')
        print(f'mean {mean:.4f}, std: {std:.4f}')
        utils.plot_histogram_and_fitted_gaussian(preds, mean, std, bins=50, savepath=f'{config.savedir}/hybrid.png')

        model = self.train_model(dataset, hidden_dims=[128], dropout=0.5)
        model.dropout_during_inference = True
    
        preds = []
        with torch.no_grad():
            for _ in range(10000):
                preds.append(model(dataset.x, dataset.edge_index)[0, dataset.y[0]])
        preds = torch.tensor(preds)
        mean = preds.mean()
        std = preds.std()
        print('MC dropout')
        print(f'mean {mean:.4f}, std: {std:.4f}')
        utils.plot_histogram_and_fitted_gaussian(preds, mean, std, bins=50, savepath=f'{config.savedir}/mc_dropout.png')

    def information_leakage(self, dataset, num_shadow_models=50, mc_dropout_samples=1000):
        hidden_dims = [2 * x for x in self.config.hidden_dim_target]
        dropout = 0.5
        train_set = datasetup.masked_subgraph(dataset, dataset.train_mask)
        test_set = datasetup.masked_subgraph(dataset, dataset.test_mask)
        node_index = torch.arange(train_set.num_nodes)[:2]
        incl_preds = []
        for _ in tqdm(range(num_shadow_models), desc='Training inclusion models'):
            incl_model = self.train_model(dataset, hidden_dims=hidden_dims, dropout=dropout)
            with torch.no_grad():
                for _ in range(mc_dropout_samples):
                    pred = incl_model(train_set.x, train_set.edge_index)[node_index, train_set.y[node_index]]
                    incl_preds.append(pred)
        incl_preds = torch.stack(incl_preds)
        incl_means = incl_preds.mean(dim=0)
        incl_stds = incl_preds.std(dim=0)

        excl_preds = []
        for node in node_index:
            preds = []
            sub_node_index = torch.cat((node_index[:node], node_index[node + 1:]))
            sub_train_set = datasetup.masked_subgraph(train_set, sub_node_index)
            sub_graph = datasetup.merge_graphs(sub_train_set, test_set)
            for _ in tqdm(range(num_shadow_models), desc='Training exclusion models'):
                excl_model = self.train_model(sub_graph, hidden_dims=hidden_dims, dropout=dropout)
                with torch.no_grad():
                    for _ in range(mc_dropout_samples):
                        pred = excl_model(train_set.x, train_set.edge_index)[node, train_set.y[node]]
                        preds.append(pred)
            preds = torch.tensor(preds)
            excl_preds.append(preds)
        excl_preds = torch.stack(excl_preds)
        excl_means = excl_preds.mean(dim=1)
        excl_stds = excl_preds.std(dim=1)

        info_leakage = gaussian_kl_divergence(incl_means, incl_stds, excl_means, excl_stds)
        return info_leakage
