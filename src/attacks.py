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
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph, degree
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from tqdm.auto import tqdm
import copy
from collections import defaultdict

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
        self.monte_carlo_masks = []
        for _ in range(config.mc_inference_samples):
            self.monte_carlo_masks.append(datasetup.random_edge_mask(self.population, 0.8))
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

    def ratio(self, target_samples, num_hops, inductive_inference, monte_carlo_masks, interp_from_out_models):
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
                    monte_carlo_masks=monte_carlo_masks,
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
                monte_carlo_masks=monte_carlo_masks,
            )
            target_confidence = F.softmax(preds, dim=1)[row_idx, target_samples.y]
        assert pr.shape == target_confidence.shape == (target_samples.num_nodes,)
        return target_confidence / pr

    def score(self, target_samples, num_hops, inductive_inference, monte_carlo_masks):
        ratioX = self.ratio(target_samples, num_hops, inductive_inference, monte_carlo_masks, interp_from_out_models=True)
        if monte_carlo_masks is not None:
            ratioZ = self.ratio(self.population, num_hops, inductive_inference, self.monte_carlo_masks, interp_from_out_models=False)
        else:
            ratioZ = self.ratio(self.population, num_hops, inductive_inference, None, interp_from_out_models=False)
        return torch.tensor([(x > ratioZ * self.gamma).float().mean().item() for x in ratioX])

    def run_attack(self, target_samples, num_hops=0, inductive_inference=True, monte_carlo_masks=None):
        target_samples.to(self.config.device)
        self.population.to(self.config.device)
        return self.score(target_samples, num_hops, inductive_inference, monte_carlo_masks)

class BayesOptimalMembershipInference:
    
    def __init__(self, target_model, graph, loss_fn, config):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config
        #self.shadow_models = []
        self.zero_hop_attacker = LiraOnline(
            target_model=target_model,
            graph=graph,
            loss_fn=loss_fn,
            config=config,
        )
        self.zero_hop_probs = self.zero_hop_attacker.run_attack(torch.arange(self.graph.num_nodes)).sigmoid()
        self.shadow_models = [shadow_model for shadow_model, _ in self.zero_hop_attacker.shadow_models]
        #self.train_shadow_models()

    def train_shadow_models(self):
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.graph.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs_shadow,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for _ in tqdm(range(config.num_shadow_models), desc=f"Training {config.num_shadow_models} shadow models for BayesOptimalMembershipInference"):
            shadow_dataset = datasetup.remasked_graph(self.graph, train_frac=config.train_frac, val_frac=config.val_frac)
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

    def masked_subgraph(self, node_mask):
        edge_index, _ = subgraph(
            subset=node_mask,
            edge_index=self.graph.edge_index,
            relabel_nodes=True,
        )
        return Data(
            x=self.graph.x[node_mask],
            edge_index=edge_index,
            y=self.graph.y[node_mask],
            num_classes=self.graph.num_classes,
        )

    def get_random_node_mask(self, frac_ones=0.5):
        return ~(torch.randint(0, int(1 / frac_ones + 1e-6), size=(self.graph.num_nodes,)).bool())

    def evaluate_mask(self, mask):
        subgraph = self.masked_subgraph(mask)
        with torch.inference_mode():
            log_p = -self.loss_fn(self.target_model(subgraph.x, subgraph.edge_index), subgraph.y)
        return log_p, subgraph

    def sample_subgraph_MCMC(self, prev_log_p, prev_mask, prev_subgraph, eps=0.1):
        u = np.random.rand()
        mask = self.get_random_node_mask(frac_ones=eps) ^ prev_mask
        log_p, subgraph = self.evaluate_mask(mask)
        crit = torch.exp(log_p - prev_log_p).item()
        if crit > u:
            #print(f'accept: {crit:.4f}')
            return log_p, mask, subgraph
        else:
            #print(f'reject: {crit:.4f}')
            return prev_log_p, prev_mask, prev_subgraph

    def sample_node_mask_zero_hop_MIA(self):
        random_ref = torch.rand(size=(self.graph.num_nodes,))
        node_mask = self.zero_hop_probs > random_ref
        return node_mask

    def log_model_posterior(self, subgraph):
        # Only loss values over the k-hop neighborhood is necessary (for a k-layer GNN)
        # but we compute loss values over all node for simplicity
        with torch.inference_mode():
            loss_term = self.loss_fn(self.target_model(subgraph.x, subgraph.edge_index), subgraph.y)
            neg_loss = torch.tensor([
                -self.loss_fn(shadow_model(subgraph.x, subgraph.edge_index), subgraph.y)
                for shadow_model in self.shadow_models
            ])
            log_Z_term = neg_loss.logsumexp(0) - np.log(self.config.num_shadow_models)
        return -loss_term - log_Z_term

    # def run_attack(self, target_node_index, prior=0.5):
    #     preds = []
    #     for node_idx in tqdm(target_node_index, total=len(target_node_index), desc="Running membership inference over all nodes"):
    #         samples = []
    #         for _ in range(self.config.num_sampled_graphs):
    #             node_mask = self.sample_node_mask_zero_hop_MIA()
    #             node_mask[node_idx] = True
    #             subgraph_in = self.masked_subgraph(node_mask)
    #             log_posterior_in = self.log_model_posterior(subgraph=subgraph_in)
    #             node_mask[node_idx] = False
    #             subgraph_out = self.masked_subgraph(node_mask)
    #             log_posterior_out = self.log_model_posterior(subgraph=subgraph_out)
    #             # No need to apply sigmoid since it is a monotone increasing function, but nice to get probabilities
    #             samples.append(torch.sigmoid(log_posterior_in - log_posterior_out + np.log(prior) - np.log(1 - prior)))
    #         samples = torch.tensor(samples)
    #         test_statistic = samples.mean()
    #         preds.append(test_statistic)
    #     preds = torch.tensor(preds)
    #     return preds
    
    def run_attack(self, target_node_index, prior=0.5):
        config = self.config
        preds = []
        samples = torch.zeros(size=(target_node_index.shape[0], config.num_sampled_graphs))
        for i in tqdm(range(config.num_sampled_graphs), desc="Monte Carlo estimation over sampled graphs"):
            node_mask = self.sample_node_mask_zero_hop_MIA()
            subgraph_in = self.masked_subgraph(node_mask)
            log_posterior_in = self.log_model_posterior(subgraph=subgraph_in)
            for j, node_idx in enumerate(target_node_index):
                node_mask[node_idx] = not node_mask[node_idx]
                subgraph_out = self.masked_subgraph(node_mask)
                log_posterior_out = self.log_model_posterior(subgraph=subgraph_out)
                if node_mask[node_idx]:
                    samples[j][i] = torch.sigmoid(log_posterior_out - log_posterior_in + np.log(prior) - np.log(1 - prior))
                else:
                    samples[j][i] = torch.sigmoid(log_posterior_in - log_posterior_out + np.log(prior) - np.log(1 - prior))
                node_mask[node_idx] = not node_mask[node_idx]
        preds = samples.mean(dim=1)
        assert preds.shape == target_node_index.shape
        return preds

class ConfidenceAttack2:

    def __init__(self, target_model, graph, config):
        self.target_model = target_model
        self.graph = graph
        self.config = config

    def run_attack(self, target_node_index):
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long)
        with torch.inference_mode():
            preds = F.softmax(self.target_model(self.graph.x, empty_edge_index), dim=1)[target_node_index, self.graph.y[target_node_index]]
        return preds

class LiraOnline:
    
    def __init__(self, target_model, graph, loss_fn, config):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config
        self.shadow_models = [] # Pairs of shadow models and its corresponding training mask
        self.train_shadow_models()
    
    def train_shadow_models(self):
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.graph.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs_shadow,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for _ in tqdm(range(config.num_shadow_models), desc=f"Training {config.num_shadow_models} shadow models for Lira online"):
            shadow_dataset = datasetup.remasked_graph(self.graph, train_frac=config.train_frac, val_frac=config.val_frac)
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
            self.shadow_models.append((shadow_model, shadow_dataset.train_mask))
    
    def query_shadow_models(self, target_node_index):
        hinges_in = defaultdict(list)
        hinges_out = defaultdict(list)
        num_target_nodes = target_node_index.shape[0]
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long)
        with torch.inference_mode():
            for shadow_model, train_mask in self.shadow_models:
                preds = shadow_model(self.graph.x[target_node_index], empty_edge_index)
                # Approximate logits of confidence values using the hinge loss.
                hinges = utils.hinge_loss(preds, self.graph.y[target_node_index])
                for idx, node_idx in enumerate(target_node_index):
                    if train_mask[node_idx]:
                        hinges_in[idx].append(hinges[idx])
                    else:
                        hinges_out[idx].append(hinges[idx])
        mean_in = torch.zeros(num_target_nodes)
        std_in = torch.zeros(num_target_nodes)
        mean_out = torch.zeros(num_target_nodes)
        std_out = torch.zeros(num_target_nodes)
        for idx, hinge_list in hinges_in.items():
            hinges = torch.tensor(hinge_list)
            mean_in[idx] = hinges.mean()
            std_in[idx] = hinges.std()
        for idx, hinge_list in hinges_out.items():
            hinges = torch.tensor(hinge_list)
            mean_out[idx] = hinges.mean()
            std_out[idx] = hinges.std()
        assert mean_in.shape == std_in.shape == mean_out.shape == std_out.shape == (num_target_nodes,)
        return mean_in, std_in, mean_out, std_out

    def run_attack(self, target_node_index):
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long)
        mean_in, std_in, mean_out, std_out = self.query_shadow_models(target_node_index)
        with torch.inference_mode():
            preds = self.target_model(self.graph.x[target_node_index], empty_edge_index)
            target_hinges = utils.hinge_loss(preds, self.graph.y[target_node_index])
        p_in = norm.logpdf(
            target_hinges.cpu().numpy(),
            loc=mean_in.cpu().numpy(),
            scale=std_in.cpu().numpy() + LiRA.EPS,
        )
        p_out = norm.logpdf(
            target_hinges.cpu().numpy(),
            loc=mean_out.cpu().numpy(),
            scale=std_out.cpu().numpy() + LiRA.EPS,
        )
        assert p_in.shape == p_out.shape == (target_node_index.shape[0],)
        return torch.tensor(p_in - p_out)
