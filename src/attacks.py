import datasetup
import evaluation
import trainer
import utils

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from tqdm.auto import tqdm
import copy

class BasicMLPAttack:

    def __init__(self, target_model, shadow_dataset, config):
        target_model.eval()
        self.target_model = target_model
        self.shadow_model = utils.fresh_model(
            model_type=config.model,
            num_features=shadow_dataset.num_features,
            hidden_dims=config.hidden_dim_target,
            num_classes=shadow_dataset.num_classes,
            dropout=config.dropout,
        )
        dims = [shadow_dataset.num_classes, *config.hidden_dim_mlp_attack, 2]
        self.attack_model = MLP(channel_list=dims, dropout=0.0)
        self.shadow_dataset = shadow_dataset
        self.config = config
        self.plot_training_results = True
        self.train_shadow_model()
        self.train_attack_model()
        self.attack_model.eval()
    
    def train_shadow_model(self):
        config = self.config
        train_config = trainer.TrainConfig(
            criterion=Accuracy(task="multiclass", num_classes=self.shadow_dataset.num_classes).to(config.device),
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=F.cross_entropy,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        train_res = trainer.train_gnn(
            model=self.shadow_model,
            dataset=self.shadow_dataset,
            config=train_config,
            inductive_split=config.inductive_split,
        )
        evaluation.evaluate_graph_training(
            model=self.shadow_model,
            dataset=self.shadow_dataset,
            criterion=train_config.criterion,
            inductive_inference=config.inductive_inference,
            training_results=train_res if self.plot_training_results else None,
            plot_title="Shadow model",
            savedir=config.savedir,
        )

    def create_attack_dataset(self):
        shadow_model = self.shadow_model
        shadow_dataset = self.shadow_dataset
        features = shadow_model(shadow_dataset.x, shadow_dataset.edge_index).cpu()
        labels = shadow_dataset.train_mask.long().cpu()
        train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, stratify=labels)
        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        return train_dataset, test_dataset

    def train_attack_model(self):
        config = self.config
        train_dataset, valid_dataset = self.create_attack_dataset()
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
        train_config = trainer.TrainConfig(
            criterion=Accuracy(task="multiclass", num_classes=2).to(config.device),
            device=config.device,
            epochs=config.epochs_mlp_attack,
            early_stopping=30,
            loss_fn=nn.CrossEntropyLoss(),
            lr=1e-3,
            weight_decay=1e-4,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        trainer.train_mlp(
            model=self.attack_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            config=train_config,
        )
    
    def run_attack(self, target_samples, num_hops=0, inductive_inference=True):
        num_target_samples = target_samples.num_nodes
        with torch.inference_mode():
            preds = evaluation.k_hop_query(
                model=self.target_model,
                dataset=target_samples,
                query_nodes=torch.arange(num_target_samples),
                num_hops=num_hops,
                inductive_split=inductive_inference,
            )
            logits = self.attack_model(preds)[:,1]
        return logits

class ImprovedMLPAttack:

    def __init__(self, target_model, population, queries, config, plot_training_results=True):
        self.config = config
        self.target_model = target_model
        self.population = population
        self.shadow_model = utils.fresh_model(
            model_type=config.model,
            num_features=population.num_features,
            hidden_dims=config.hidden_dim_target,
            num_classes=population.num_classes,
            dropout=config.dropout,
        )
        self.plot_training_results = plot_training_results
        self.queries = queries
        dims = [population.num_classes * len(queries), *config.hidden_dim_mlp_attack, 2]
        self.attack_model = MLP(channel_list=dims, dropout=0.0)
        self.train_shadow_model()
        self.train_attack_model()
        self.attack_model.eval()

    def train_shadow_model(self):
        config = self.config
        train_config = trainer.TrainConfig(
            criterion=Accuracy(task="multiclass", num_classes=self.population.num_classes).to(config.device),
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=F.cross_entropy,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        train_res = trainer.train_gnn(
            model=self.shadow_model,
            dataset=self.population,
            config=train_config,
            inductive_split=config.inductive_split,
        )
        evaluation.evaluate_graph_training(
            model=self.shadow_model,
            dataset=self.population,
            criterion=train_config.criterion,
            inductive_inference=config.inductive_inference,
            training_results=train_res if self.plot_training_results else None,
            plot_title="Shadow model",
            savedir=config.savedir,
        )

    def make_attack_dataset(self):
        features = []
        row_idx = torch.arange(self.population.num_nodes)
        for num_hops in self.queries:
            preds = evaluation.k_hop_query(
                model=self.shadow_model,
                dataset=self.population,
                query_nodes=row_idx,
                num_hops=num_hops,
                inductive_split=self.config.inductive_inference,
            )
            features.append(preds)
        features = torch.cat(features, dim=1).cpu()
        labels = self.population.train_mask.long().cpu()
        train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, stratify=labels)
        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        return train_dataset, test_dataset

    def train_attack_model(self):
        config = self.config
        train_dataset, valid_dataset = self.make_attack_dataset()
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
        train_config = trainer.TrainConfig(
            criterion=Accuracy(task="multiclass", num_classes=2).to(config.device),
            device=config.device,
            epochs=config.epochs_mlp_attack,
            early_stopping=30,
            loss_fn=nn.CrossEntropyLoss(),
            lr=1e-3,
            weight_decay=1e-4,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        trainer.train_mlp(
            model=self.attack_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            config=train_config,
        )

    def run_attack(self, target_samples, num_hops=None, inductive_inference=True):
        # num_hops not used, attack always use both 0-hop and 2-hop queries
        row_idx = torch.arange(target_samples.num_nodes)
        with torch.inference_mode():
            features = []
            for num_hops in self.queries:
                preds = evaluation.k_hop_query(
                    model=self.target_model,
                    dataset=target_samples,
                    query_nodes=row_idx,
                    num_hops=num_hops,
                    inductive_split=inductive_inference,
                )
                features.append(preds)
            features = torch.cat(features, dim=1)
            logits = self.attack_model(features)[:,1]
        return logits

class ConfidenceAttack:
    
    def __init__(self, target_model, config):
        target_model.eval()
        self.target_model = target_model
        self.config = config
    
    def run_attack(self, target_samples, num_hops=0, inductive_inference=True, monte_carlo_masks=None):
        row_idx = torch.arange(target_samples.num_nodes)
        with torch.inference_mode():
            preds = evaluation.k_hop_query(
                model=self.target_model,
                dataset=target_samples,
                query_nodes=row_idx,
                num_hops=num_hops,
                inductive_split=inductive_inference,
                monte_carlo_masks=monte_carlo_masks,
            )
            confidences = preds[row_idx, target_samples.y] # Unnormalized for numerical stability
        return confidences

class LiRA:
    '''
    The (offline) likelihood ratio attack from "Membership Inference Attacks From First Principles"
    '''
    EPS = 1e-8

    def __init__(self, target_model, population, config):
        target_model.eval()
        self.target_model = target_model
        self.shadow_models = []
        self.population = population # Should not contain target samples.
        self.config = config
        self.train_shadow_models()

    def train_shadow_models(self):
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.population.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs_shadow,
            early_stopping=config.early_stopping,
            loss_fn=F.cross_entropy,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for _ in tqdm(range(config.num_shadow_models), desc=f"Training {config.num_shadow_models} shadow models for LiRA"):
            shadow_dataset = datasetup.remasked_graph(self.population, train_frac=config.train_frac, val_frac=config.val_frac, stratify=self.population.y)
            shadow_model = utils.fresh_model(
                model_type=config.model,
                num_features=shadow_dataset.num_features,
                hidden_dims=config.hidden_dim_target,
                num_classes=shadow_dataset.num_classes,
                dropout=config.dropout,
            )
            _ = trainer.train_gnn(
                model=shadow_model,
                dataset=shadow_dataset,
                config=train_config,
                disable_tqdm=True,
                inductive_split=config.inductive_split,
            )
            shadow_model.eval()
            self.shadow_models.append(shadow_model)
    
    def get_mean_and_std(self, target_samples, num_hops, inductive_inference, monte_carlo_masks):
        hinges = []
        for shadow_model in self.shadow_models:
            with torch.inference_mode():
                preds = evaluation.k_hop_query(
                    model=shadow_model,
                    dataset=target_samples,
                    query_nodes=torch.arange(target_samples.num_nodes),
                    num_hops=num_hops,
                    inductive_split=inductive_inference,
                    monte_carlo_masks=monte_carlo_masks,
                )
                # Approximate logits of confidence values using the hinge loss.
                hinges.append(utils.hinge_loss(preds, target_samples.y))
        hinges = torch.stack(hinges)
        assert hinges.shape == (len(self.shadow_models), target_samples.num_nodes)
        means = hinges.mean(dim=0)
        stds = hinges.std(dim=0)
        utils.plot_histogram_and_fitted_gaussian(
            x=hinges[:,0].cpu().numpy(),
            mean=means[0].cpu().numpy(),
            std=stds[0].cpu().numpy(),
            bins=min(self.config.num_shadow_models // 4, 50),
            savepath="./results/LiRA_gaussian_fit_histogram.png",
        )
        return means, stds

    def run_attack(self, target_samples, num_hops=0, inductive_inference=True, monte_carlo_masks=None):
        target_samples.to(self.config.device)
        means, stds = self.get_mean_and_std(target_samples, num_hops, inductive_inference, monte_carlo_masks)
        with torch.inference_mode():
            preds = evaluation.k_hop_query(
                model=self.target_model,
                dataset=target_samples,
                query_nodes=torch.arange(target_samples.num_nodes),
                num_hops=num_hops,
                inductive_split=inductive_inference,
                monte_carlo_masks=monte_carlo_masks,
            )
            target_hinges = utils.hinge_loss(preds, target_samples.y)

        # In offline LiRA the test statistic is Lambda = 1 - P(Z > conf_target), where Z is a sample from
        # a normal distribution with mean and variance given by the shadow models confidences.
        # We normalize the target confidence and compute the test statistic Lambda' = P(Z < x), Z ~ Normal(0, 1)
        # For numerical stability, compute the log CDF.
        preds = norm.logcdf(
            target_hinges.cpu().numpy(),
            loc=means.cpu().numpy(),
            scale=stds.cpu().numpy() + LiRA.EPS,
        )
        return torch.tensor(preds)

class RMIA:
    '''
    The offline RMIA attack from "Low-Cost High-Power Membership Inference Attacks".
    '''

    def __init__(self, target_model, population, config):
        target_model.eval()
        self.target_model = target_model
        self.population = population
        self.config = config
        self.shadow_models = []
        self.shadow_datasets = []
        self.out_size = population.num_nodes
        self.gamma = config.rmia_gamma
        self.partition_population()
        self.train_shadow_models()
        self.offline_a = self.select_offline_a()
        print("offline_a:", self.offline_a)

    def partition_population(self, unbiased_in_expectation=True):
        '''
        Partition the training nodes for the shadow models such that each model is trained on 50% of the population nodes.
        '''
        config = self.config
        if unbiased_in_expectation:
            for _ in range(config.num_shadow_models):
                shadow_dataset = datasetup.remasked_graph(self.population, train_frac=0.5, val_frac=0.0)
                self.shadow_datasets.append(shadow_dataset)
        else:
            shadow_indices = [[] for _ in range(config.num_shadow_models)]
            node_indices = np.arange(self.population.num_nodes)
            for node_index in node_indices:
                chosen_models = np.random.choice(config.num_shadow_models, config.num_shadow_models // 2, replace=False)
                for model_index in chosen_models:
                    shadow_indices[model_index].append(node_index)
            for node_index in shadow_indices:
                node_index = torch.tensor(node_index, dtype=torch.long)
                shadow_dataset = copy.deepcopy(self.population)
                train_mask = torch.zeros(shadow_dataset.num_nodes, dtype=torch.bool)
                train_mask[node_index] = True
                test_mask = torch.zeros(shadow_dataset.num_nodes, dtype=torch.bool)
                test_mask[~node_index] = True
                val_mask = torch.zeros(shadow_dataset.num_nodes, dtype=torch.bool)
                shadow_dataset.train_mask = train_mask
                shadow_dataset.test_mask = test_mask
                shadow_dataset.val_mask = val_mask
                shadow_dataset.inductive_mask = datasetup.train_split_interconnection_mask(shadow_dataset)
                self.shadow_datasets.append(shadow_dataset)

    def train_shadow_models(self):
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.population.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs_shadow,
            early_stopping=0,
            loss_fn=F.cross_entropy,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for i in tqdm(range(config.num_shadow_models), desc=f"Training {config.num_shadow_models} out models for RMIA"):
            shadow_dataset = self.shadow_datasets[i]
            shadow_model = utils.fresh_model(
                model_type=config.model,
                num_features=shadow_dataset.num_features,
                hidden_dims=config.hidden_dim_target,
                num_classes=shadow_dataset.num_classes,
                dropout=config.dropout,
            )
            _ = trainer.train_gnn(
                model=shadow_model,
                dataset=shadow_dataset,
                config=train_config,
                disable_tqdm=True,
                inductive_split=config.inductive_split,
            )
            shadow_model.eval()
            self.shadow_models.append(shadow_model)

    def select_offline_a(self):
        target_index, shadow_index = np.random.choice(self.config.num_shadow_models, 2, replace=False)
        sim_target_model, sim_target_dataset = self.shadow_models[target_index], self.shadow_datasets[target_index]
        sim_shadow_model, sim_shadow_dataset = self.shadow_models[shadow_index], self.shadow_datasets[shadow_index]
        # Assumes the nodes are labeled the same in sim_target_dataset and sim_shadow_dataset.
        target_samples = datasetup.masked_subgraph(sim_target_dataset, ~(sim_target_dataset.val_mask | sim_shadow_dataset.train_mask))
        population = datasetup.masked_subgraph(sim_shadow_dataset, ~(sim_target_dataset.train_mask))
        best_auroc = 0
        best_offline_a = 0.0
        for offline_a in np.linspace(0, 1, 11):
            sim_shadow_model.eval()
            confidences = []
            with torch.inference_mode():
                for model, dataset in (sim_shadow_model, target_samples), (sim_target_model, target_samples), (sim_shadow_model, population), (sim_target_model, population):
                    row_idx = torch.arange(dataset.num_nodes)
                    preds = evaluation.k_hop_query(
                        model=model,
                        dataset=dataset,
                        query_nodes=row_idx,
                        num_hops=0,
                        inductive_split=self.config.inductive_inference,
                    )
                    confidences.append(F.softmax(preds, dim=1)[row_idx, dataset.y])

            ratioX = confidences[1] / (0.5 * ((offline_a + 1) * confidences[0] + 1 - offline_a))
            ratioZ = confidences[3] / confidences[2]
            score = torch.tensor([(x > ratioZ * self.gamma).float().mean().item() for x in ratioX])
            auroc = roc_auc_score(y_true=target_samples.train_mask, y_score=score)
            if auroc > best_auroc:
                best_auroc = auroc
                best_offline_a = offline_a
        return best_offline_a

    def ratio(self, target_samples, num_hops, inductive_inference, interp_from_out_models):
        row_idx = torch.arange(target_samples.num_nodes)
        shadow_confidences = []
        for shadow_model in self.shadow_models:
            with torch.inference_mode():
                preds = evaluation.k_hop_query(
                    model=shadow_model,
                    dataset=target_samples,
                    query_nodes=row_idx,
                    num_hops=num_hops,
                    inductive_split=inductive_inference,
                )
                shadow_confidences.append(F.softmax(preds, dim=1)[row_idx, target_samples.y])
        shadow_confidences = torch.stack(shadow_confidences)
        if interp_from_out_models:
            pr_out = shadow_confidences.mean(dim=0)
            # Heuristic to approximate the average of in and out, from out only.
            pr = 0.5 * ((self.offline_a + 1) * pr_out + 1 - self.offline_a)
        else:
            pr = shadow_confidences.mean(dim=0)

        with torch.inference_mode():
            preds = evaluation.k_hop_query(
                model=self.target_model,
                dataset=target_samples,
                query_nodes=row_idx,
                num_hops=num_hops,
                inductive_split=inductive_inference,
            )
            target_confidence = F.softmax(preds, dim=1)[row_idx, target_samples.y]
        assert pr.shape == target_confidence.shape == (target_samples.num_nodes,)
        return target_confidence / pr

    def score(self, target_samples, num_hops, inductive_inference):
        ratioX = self.ratio(target_samples, num_hops, inductive_inference, interp_from_out_models=True)
        ratioZ = self.ratio(self.population, num_hops, inductive_inference, interp_from_out_models=False)
        return torch.tensor([(x > ratioZ * self.gamma).float().mean().item() for x in ratioX])

    def run_attack(self, target_samples, num_hops=0, inductive_inference=True):
        target_samples.to(self.config.device)
        self.population.to(self.config.device)
        return self.score(target_samples, num_hops, inductive_inference)
