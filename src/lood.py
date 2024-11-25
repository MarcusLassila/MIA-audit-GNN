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
        edge_index = torch.tensor([[],[]], dtype=torch.bool)
        for _ in tqdm(range(5000), desc="Training shadow models"):
            model = self.train_model(dataset, hidden_dims=[64], dropout=0.0)
            with torch.inference_mode():
                preds.append(model(dataset.x, edge_index)[0, dataset.y[0]])
        preds = torch.tensor(preds)
        mean = preds.mean()
        std = preds.std()
        print('Retrain')
        print(f'mean {mean:.4f}, std: {std:.4f}')
        utils.plot_histogram_and_fitted_gaussian(preds, mean, std, bins=50, savepath=f'{config.savedir}/retrain.png')

        preds = []
        edge_index = torch.tensor([[],[]], dtype=torch.bool)
        for _ in tqdm(range(50), desc="Training shadow models"):
            model = self.train_model(dataset, hidden_dims=[64], dropout=0.0)
            with torch.inference_mode():
                preds.append(model(dataset.x, edge_index)[0, dataset.y[0]])
        preds = torch.tensor(preds)
        mean = preds.mean()
        std = preds.std()
        print('Retrain budget')
        print(f'mean {mean:.4f}, std: {std:.4f}')
        utils.plot_histogram_and_fitted_gaussian(preds, mean, std, bins=50, savepath=f'{config.savedir}/retrain_budget.png')

        preds = []
        for _ in tqdm(range(50), desc="Training shadow models"):
            model = self.train_model(dataset, hidden_dims=[128], dropout=0.5)
            model.dropout_during_inference = True
            with torch.inference_mode():
                for _ in range(1000):
                    preds.append(model(dataset.x, edge_index)[0, dataset.y[0]])
        preds = torch.tensor(preds)
        mean = preds.mean()
        std = preds.std()
        print('Hybrid')
        print(f'mean {mean:.4f}, std: {std:.4f}')
        utils.plot_histogram_and_fitted_gaussian(preds, mean, std, bins=50, savepath=f'{config.savedir}/hybrid.png')

        model = self.train_model(dataset, hidden_dims=[128], dropout=0.5)
        model.dropout_during_inference = True
    
        preds = []
        with torch.inference_mode():
            for _ in range(10000):
                preds.append(model(dataset.x, edge_index)[0, dataset.y[0]])
        preds = torch.tensor(preds)
        mean = preds.mean()
        std = preds.std()
        print('MC dropout')
        print(f'mean {mean:.4f}, std: {std:.4f}')
        utils.plot_histogram_and_fitted_gaussian(preds, mean, std, bins=50, savepath=f'{config.savedir}/mc_dropout.png')

    def information_leakage(self, dataset, node_index, num_shadow_models=50):
        hidden_dims = self.config.hidden_dim_target
        dropout = 0.5
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long)
        incl_preds = []
        for _ in tqdm(range(num_shadow_models), desc='Training inclusion models'):
            incl_model = self.train_model(dataset, hidden_dims=hidden_dims, dropout=dropout)
            with torch.inference_mode():
                pred = incl_model(dataset.x, empty_edge_index)[node_index, dataset.y[node_index]]
                incl_preds.append(pred)
        incl_preds = torch.stack(incl_preds)
        incl_means = incl_preds.mean(dim=0)
        incl_stds = incl_preds.std(dim=0)
        assert incl_preds.shape == (num_shadow_models, node_index.shape[0])
        assert incl_means.shape == node_index.shape

        excl_preds = []
        for node in node_index:
            preds = []
            mask = torch.ones(dataset.num_nodes, dtype=torch.bool)
            mask[node] = False
            sub_dataset= datasetup.masked_subgraph(dataset, mask)
            for _ in tqdm(range(num_shadow_models), desc='Training exclusion models'):
                excl_model = self.train_model(sub_dataset, hidden_dims=hidden_dims, dropout=dropout)
                with torch.inference_mode():
                    pred = excl_model(sub_dataset.x, empty_edge_index)[node, dataset.y[node]]
                    preds.append(pred)
            preds = torch.tensor(preds)
            excl_preds.append(preds)
        excl_preds = torch.stack(excl_preds)
        excl_means = excl_preds.mean(dim=1)
        excl_stds = excl_preds.std(dim=1)
        assert excl_preds.shape == (node_index.shape[0], num_shadow_models)
        assert excl_means.shape == node_index.shape

        info_leakage = gaussian_kl_divergence(incl_means, incl_stds, excl_means, excl_stds)
        return info_leakage
