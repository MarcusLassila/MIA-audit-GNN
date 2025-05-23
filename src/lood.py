import datasetup
import evaluation
import trainer
import utils

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import subgraph, k_hop_subgraph
from torchmetrics import Accuracy
from tqdm.auto import tqdm

def gaussian_kl_divergence(mean_0, std_0, mean_1, std_1):
    var_0 = std_0 ** 2
    var_1 = std_1 ** 2
    return torch.log(std_1 / std_0) - 0.5 + (var_0 + (mean_0 - mean_1) ** 2) / (2 * var_1)

def multi_gaussian_kl_divergence(mean_0, cov_0, mean_1, cov_1):
    # Better implementation with Cholesky decomposition?
    inv_cov_1 = torch.linalg.inv(cov_1)
    det_cov_0 = torch.linalg.det(cov_0)
    det_cov_1 = torch.linalg.det(cov_1)
    trace = torch.trace(inv_cov_1 @ cov_0)
    dim = mean_0.shape[0]
    mu = mean_1 - mean_0
    quad = mu.t() @ inv_cov_1 @ mu
    log = torch.log(det_cov_1 / det_cov_0)
    return 0.5 * (trace - dim + quad + log)

class LOOD:
    
    def __init__(self, config):
        self.config = config

    def train_model(self, dataset, hidden_dims=None, dropout=None, disable_tqdm=True):
        config = self.config
        hidden_dims = config.hidden_dim if hidden_dims is None else hidden_dims
        dropout = config.dropout if dropout is None else dropout
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
            epochs=config.epochs,
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
        for _ in tqdm(range(10000), desc="Training shadow models"):
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
        for _ in tqdm(range(256), desc="Training shadow models"):
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

    def information_leakage(self, dataset, node_index, num_hops=0, num_shadow_models=256):
        row_idx = torch.arange(node_index.shape[0])
        in_confs = []
        in_match = []
        for _ in tqdm(range(num_shadow_models), desc='Training inclusion models'):
            in_model = self.train_model(dataset)
            pred = utils.k_hop_query(
                model=in_model,
                dataset=dataset,
                query_nodes=node_index,
                num_hops=num_hops,
                inductive_split=True,
            )
            conf = pred[row_idx, dataset.y[node_index]]
            match = (pred.argmax(dim=1) == dataset.y[node_index]).float()
            in_confs.append(conf)
            in_match.append(match)
        in_confs = torch.stack(in_confs)
        in_memo = torch.stack(in_match).mean(dim=0)
        in_means = in_confs.mean(dim=0)
        in_stds = in_confs.std(dim=0)
        assert in_confs.shape == (num_shadow_models, node_index.shape[0])
        assert in_means.shape == node_index.shape
        assert in_memo.shape == node_index.shape

        out_confs = []
        out_match = []
        for i in tqdm(row_idx, desc="Training exclusion models"):
            node = node_index[i].unsqueeze(dim=0)
            confs = []
            match = []
            mask = torch.ones(dataset.num_nodes, dtype=torch.bool)
            # sub_node_index, _, _, _ = k_hop_subgraph(
            #     node_idx=[node],
            #     num_hops=num_hops,
            #     edge_index=dataset.edge_index,
            #     num_nodes=dataset.num_nodes,
            #     relabel_nodes=False,
            # )
            sub_node_index = node
            mask[sub_node_index] = False
            sub_dataset= datasetup.masked_subgraph(dataset, mask, include_train_masks=True)
            assert sub_dataset.num_nodes + sub_node_index.shape[0] == dataset.num_nodes
            for _ in range(num_shadow_models):
                out_model = self.train_model(sub_dataset)
                pred = utils.k_hop_query(
                    model=out_model,
                    dataset=dataset,
                    query_nodes=node,
                    num_hops=num_hops,
                    inductive_split=True,
                ).squeeze()
                assert pred.shape == (dataset.num_classes,)
                match.append((pred.argmax() == dataset.y[node]).float())
                confs.append(pred[dataset.y[node]])
            confs = torch.tensor(confs)
            match = torch.tensor(match)
            out_confs.append(confs)
            out_match.append(match)
        out_confs = torch.stack(out_confs)
        out_memo = torch.stack(out_match).mean(dim=1)
        out_means = out_confs.mean(dim=1)
        out_stds = out_confs.std(dim=1)
        assert out_confs.shape == (node_index.shape[0], num_shadow_models)
        assert out_means.shape == node_index.shape
        assert out_memo.shape == node_index.shape

        kl_leakage = gaussian_kl_divergence(in_means, in_stds, out_means, out_stds)
        memo = in_memo - out_memo
        return kl_leakage, memo

    def information_leakage_full(self, dataset, node_index, num_hops=0, num_shadow_models=256):

        in_models = []
        for _ in tqdm(range(num_shadow_models), desc='Training inclusion models'):
            in_models.append(self.train_model(dataset))

        info_leakage = []
        for i in tqdm(range(node_index.shape[0]), desc="Training exclusion models"):
            node = node_index[i].unsqueeze(dim=0)
            in_preds = []
            out_preds = []
            mask = torch.ones(dataset.num_nodes, dtype=torch.bool)
            # sub_node_index, _, _, _ = k_hop_subgraph(
            #     node_idx=node,
            #     num_hops=num_hops,
            #     edge_index=dataset.edge_index,
            #     num_nodes=dataset.num_nodes,
            #     relabel_nodes=False,
            # )
            sub_node_index = node
            mask[sub_node_index] = False
            sub_dataset= datasetup.masked_subgraph(dataset, mask, include_train_masks=True)
            assert sub_dataset.num_nodes + sub_node_index.shape[0] == dataset.num_nodes
            for in_model in in_models:
                out_model = self.train_model(sub_dataset)
                in_pred = utils.k_hop_query(
                    model=in_model,
                    dataset=dataset,
                    query_nodes=node,
                    num_hops=num_hops,
                    inductive_split=True,
                ).squeeze()
                out_pred = utils.k_hop_query(
                    model=out_model,
                    dataset=dataset,
                    query_nodes=node,
                    num_hops=num_hops,
                    inductive_split=True,
                ).squeeze()
                assert in_pred.shape == out_pred.shape == (dataset.num_classes,)
                in_preds.append(in_pred)
                out_preds.append(out_pred)
            in_preds = torch.stack(in_preds)
            out_preds = torch.stack(out_preds)
            in_means = in_preds.mean(dim=0)
            out_means = out_preds.mean(dim=0)
            in_z = in_preds - in_means
            out_z = out_preds - out_means
            in_cov = torch.matmul(in_z.t(), in_z) / (num_shadow_models - 1)
            out_cov = torch.matmul(out_z.t(), out_z) / (num_shadow_models - 1)
            assert in_means.shape == (dataset.num_classes,)
            assert in_cov.shape == (dataset.num_classes, dataset.num_classes)
            assert out_means.shape == (dataset.num_classes,)
            assert out_cov.shape == (dataset.num_classes, dataset.num_classes)
            info_leakage.append(multi_gaussian_kl_divergence(in_means, in_cov, out_means, out_cov))

        return torch.tensor(info_leakage)
