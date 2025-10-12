import datasetup
import evaluation
import models
import trainer
import utils

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t as t_dist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch_geometric.nn import MLP
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph, degree, index_to_mask, mask_to_index
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from tqdm.auto import tqdm
import copy
from collections import defaultdict

from laplace import Laplace

class MLPAttack:

    def __init__(self, target_model, graph, loss_fn, config, shadow_models=None):
        self.config = config
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.queries = config.mlp_attack_queries
        if hasattr(config, 'use_xmlp') and config.use_xmlp:
            self.attack_model = models.XMLP(
                concats=len(self.queries),
                in_dim=graph.num_classes,
                hidden_dims=config.hidden_dim_mlp,
                out_dim=2,
            )
            self.name = 'XMLP attack'
        else:
            self.attack_model = models.MLP(
                in_dim=graph.num_classes*len(self.queries),
                hidden_dims=config.hidden_dim_mlp,
                out_dim=2,
            )
            self.name = 'MLP attack'
        if shadow_models is None:
            self.shadow_models = trainer.train_shadow_models(graph, loss_fn, config)
        else:
            self.shadow_models = shadow_models
        self.train_attack_model()

    def make_attack_dataset(self):
        features = []
        labels = []
        for shadow_model, train_mask in self.shadow_models:
            shadow_graph = datasetup.remasked_graph(self.graph, train_mask, mutate=False)
            feat = []
            row_idx = torch.arange(shadow_graph.num_nodes)
            for num_hops in self.queries:
                preds = utils.k_hop_query(
                    model=shadow_model,
                    dataset=shadow_graph,
                    query_nodes=row_idx,
                    num_hops=num_hops,
                    inductive_split=True,
                    edge_dropout=self.config.edge_dropout,
                )
                feat.append(preds)
            feat = torch.cat(feat, dim=1).cpu()
            lbl = shadow_graph.train_mask.long().cpu()
            features.append(feat)
            labels.append(lbl)
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
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
            epochs=config.epochs_mlp,
            early_stopping=50,
            loss_fn=nn.CrossEntropyLoss(),
            lr=1e-3,
            weight_decay=1e-4,
            optimizer=torch.optim.Adam,
        )
        train_res = trainer.train_mlp(
            model=self.attack_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            config=train_config,
        )
        utils.plot_training_results(
            train_res,
            name=f'{self.name} attack training result',
            savedir='temp_results/MLP_attack_train_results'
        )
        self.attack_model.eval()

    @torch.inference_mode()
    def run_attack(self, target_node_index):
        # num_hops not used, attack always use both 0-hop and 2-hop queries
        features = []
        for num_hops in self.queries:
            preds = utils.k_hop_query(
                model=self.target_model,
                dataset=self.graph,
                query_nodes=target_node_index,
                num_hops=num_hops,
                edge_dropout=self.config.edge_dropout,
            )
            features.append(preds)
        features = torch.cat(features, dim=1)
        logits = self.attack_model(features)[:,1]
        return logits

class G_BASE:

    class SamplingState:
        '''State object when sampling graphs'''

        def __init__(self, outer_cls, score_dim, config, strategy='model-independent'):
            self.score = torch.zeros(size=score_dim)
            self.precisions = []
            self.recalls = []
            self.strategy = strategy
            if strategy == 'metropolis':
                self.burn_in_iterations = config.burn_in_iterations
                self.sampling_iterations = config.sampling_iterations
                self.mask = outer_cls.sample_random_node_mask(frac_ones=0.5)
                self.mask, self.log_p, self.subgraph = outer_cls.metropolis_propose(self, flip_frac=0.0)
                accepts = 0
                print(f'Log model posterior before burn-in: {self.log_p}')
                for _ in tqdm(range(self.burn_in_iterations), desc='metropolis burn-in'):
                    accepts += outer_cls.metropolis_update_step(self)
                print(f'Log model posterior after burn-in: {self.log_p}')
                print(f'accept rate: {accepts / self.burn_in_iterations:.5f}')
                self.accept_rates = []

        def metropolis_update_state(self, mask, log_p, subgraph):
            assert self.strategy == 'metropolis'
            self.mask = mask
            self.log_p = log_p
            self.subgraph = subgraph

    def __init__(self, target_model, graph, loss_fn, config, shadow_models=None):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config
        self.offline = config.offline
        if hasattr(config, 'specialized_shadow_models'):
            self.specialized_shadow_models = config.specialized_shadow_models
        else:
            self.specialized_shadow_models = False
        try:
            self.prior = config.prior
        except AttributeError:
            print('No prior specified. Using default value 0.5')
            self.prior = 0.5
        if config.sampling_strategy == 'mia':
            self.zero_hop_attacker = BASE(
                target_model=target_model,
                graph=graph,
                loss_fn=loss_fn,
                config=config,
                shadow_models=shadow_models,
                # BASE is only used here to sample neighborhoods for the target nodes.
                # When using online BASE we implicitly assume the shadow models are only
                # offline with respect to the target node, and not any of its neighbors.
                offline=False,
            )
            self.zero_hop_probs = self.zero_hop_attacker.run_attack(torch.arange(self.graph.num_nodes))
            assert torch.all(self.zero_hop_probs >= 0.0)
            assert torch.all(self.zero_hop_probs <= 1.0)
            self.shadow_models = self.zero_hop_attacker.shadow_models
        elif self.specialized_shadow_models:
            self.shadow_models = []
        elif shadow_models is None:
            self.shadow_models = trainer.train_shadow_models(self.graph, self.loss_fn, self.config)
        else:
            self.shadow_models = shadow_models
        try:
            self.threshold_scale_factor = config.threshold_scale_factor
        except AttributeError:
            self.threshold_scale_factor = 1.0
        print(f'G-BASE threshold scale facot: {self.threshold_scale_factor}')

    def train_shadow_models(self, target_idx, train_mask):
        config = self.config
        train_mask = train_mask.clone()
        shadow_models = []
        criterion = Accuracy(task="multiclass", num_classes=self.graph.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for i in range(config.num_shadow_models):
            train_mask[target_idx] = i % 2 == 0
            shadow_dataset = datasetup.remasked_graph(self.graph, train_mask)
            shadow_model = utils.fresh_model(
                model_type=config.model,
                num_features=shadow_dataset.num_features,
                hidden_dims=config.hidden_dim,
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
            shadow_models.append((shadow_model, train_mask))
        return shadow_models

    def metropolis_propose(self, sampling_state, flip_frac):
        mask = self.sample_random_node_mask(frac_ones=flip_frac) ^ sampling_state.mask
        subgraph = datasetup.masked_subgraph(self.graph, mask)
        neg_loss_term = self.neg_loss(self.target_model, subgraph)
        shadow_losses = torch.tensor([
            self.neg_loss(shadow_model, subgraph)
            for shadow_model, _ in self.shadow_models
        ])
        log_Z_term = shadow_losses.logsumexp(0) - np.log(shadow_losses.shape[0])
        log_p = neg_loss_term - self.threshold_scale_factor * log_Z_term
        return mask, log_p, subgraph

    def metropolis_update_step(self, sampling_state):
        u = np.random.rand()
        mask, log_p, subgraph = self.metropolis_propose(sampling_state, self.config.flip_frac)
        crit = torch.exp(log_p - sampling_state.log_p).item()
        if crit > u:
            sampling_state.metropolis_update_state(mask, log_p, subgraph)
            return 1
        return 0

    def gibbs_sampling(self, num_passes=1):
        mask = self.sample_random_node_mask(frac_ones=self.prior)
        for _ in range(num_passes):
            for node_idx in tqdm(torch.randperm(self.graph.num_nodes, device=self.config.device), total=self.graph.num_nodes, desc='gibbs sampling'):
                mask[node_idx] = True
                in_subgraph, mapped_center_index = self.local_subgraph(
                    node_mask=mask,
                    center_idx=node_idx,
                    num_hops=self.shadow_models[0][0].num_layers**2,
                )
                out_subgraph = datasetup.remove_node(in_subgraph, mapped_center_index)
                p = self.score(in_subgraph=in_subgraph, out_subgraph=out_subgraph, target_idx=None)
                mask[node_idx] = p > np.random.rand()
        return mask

    def sample_node_mask_zero_hop_MIA(self, prob_scaling=1.0):
        random_ref = torch.rand(self.graph.num_nodes).to(self.config.device)
        sharp_probs = torch.sigmoid(prob_scaling * self.zero_hop_probs.logit())
        return sharp_probs > random_ref

    def sample_random_node_mask(self, frac_ones=None):
        if frac_ones is None:
            frac_ones = self.prior
        mask = torch.rand(self.graph.num_nodes) < frac_ones
        return mask.to(self.config.device)

    @torch.inference_mode()
    def neg_loss(self, model, graph):
        return -F.cross_entropy(model(graph.x, graph.edge_index), graph.y, reduction='sum')

    def score(self, in_subgraph, out_subgraph, target_idx=None):
        target_loss_diff = self.neg_loss(self.target_model, in_subgraph) - self.neg_loss(self.target_model, out_subgraph)
        shadow_loss_diff = torch.tensor([
            self.neg_loss(shadow_model, in_subgraph) - self.neg_loss(shadow_model, out_subgraph)
            for shadow_model, train_index in self.shadow_models
            # Use shadow model if online, or if offline and target is not in shadow dataset
            if not self.offline or target_idx is None or not train_index[target_idx]
        ])
        threshold = shadow_loss_diff.logsumexp(0) - np.log(shadow_loss_diff.shape[0])
        score = target_loss_diff - self.threshold_scale_factor * threshold
        score += np.log(self.config.prior) - np.log(1 - self.config.prior)
        return score.sigmoid()

    def run_attack(self, target_node_index):
        config = self.config
        sampling_state = G_BASE.SamplingState(
            outer_cls=self,
            score_dim=(config.num_sampled_graphs, target_node_index.shape[0]),
            config=config,
            strategy=config.sampling_strategy,
        )
        for i in range(config.num_sampled_graphs):
            self.update_scores(
                sample_idx=i,
                target_node_index=target_node_index,
                sampling_state=sampling_state,
            )
        precision = torch.tensor(sampling_state.precisions)
        recall = torch.tensor(sampling_state.recalls)
        if config.num_sampled_graphs > 1:
            print(f'Precision of sampled graphs: {precision.mean():.5f} ({precision.std():.5f})')
            print(f'Recall of sampled graphs: {recall.mean():.5f} ({recall.std():.5f})')
        else:
            print(f'Precision of sampled graphs: {precision.mean():.5f}')
            print(f'Recall of sampled graphs: {recall.mean():.5f}')
        if sampling_state.strategy == 'metropolis':
            accept_rate = torch.tensor(sampling_state.accept_rates)
            print(f'Accept rate: {accept_rate.mean():.5f} ({accept_rate.std() if accept_rate.shape[0] > 1 else 0.0:.5f})')
        preds = sampling_state.score.mean(dim=0)
        assert preds.shape == target_node_index.shape
        return preds

    def update_scores(self, sample_idx, target_node_index, sampling_state):
        config = self.config
        match config.sampling_strategy:
            case 'model-independent':
                try:
                    frac_ones = config.frac_ones_sampled_node_mask
                except AttributeError:
                    frac_ones = 0.5
                node_mask = self.sample_random_node_mask(frac_ones=frac_ones)
            case 'mia':
                try:
                    prob_scaling = config.mia_prob_scaling
                except AttributeError:
                    prob_scaling = 1.0
                node_mask = self.sample_node_mask_zero_hop_MIA(prob_scaling=prob_scaling)
            case 'metropolis':
                accepts = 0
                for _ in range(sampling_state.sampling_iterations):
                    accepts += self.metropolis_update_step(sampling_state)
                accept_rate = accepts / sampling_state.sampling_iterations
                sampling_state.accept_rates.append(accept_rate)
                node_mask = sampling_state.mask
            case 'gibbs':
                node_mask = self.gibbs_sampling(num_passes=config.num_passes)
            case 'ground-truth':
                # ground-truth + specialized_shadow_models = leave one out attack
                node_mask = self.graph.train_mask.clone()
            case _:
                raise ValueError(f'Unsupported sampling strategy: {config.sampling_strategy}')

        num_members = self.graph.train_mask.sum().item()
        num_nodes = node_mask.sum().item()
        precision = (node_mask & self.graph.train_mask).sum() / num_nodes
        recall = (node_mask & self.graph.train_mask).sum() / num_members
        sampling_state.precisions.append(precision)
        sampling_state.recalls.append(recall)

        desc = f'Inference over target nodes for graph sample {sample_idx + 1}/{config.num_sampled_graphs}'
        for i, node_idx in tqdm(enumerate(target_node_index), total=len(target_node_index), desc=desc):
            if self.specialized_shadow_models:
               self.shadow_models = self.train_shadow_models(node_idx, node_mask)
            subgraph_in, center_idx = self.local_subgraph(
                node_mask=node_mask,
                center_idx=node_idx,
                num_hops=self.shadow_models[0][0].num_layers**2,
            )
            subgraph_out = datasetup.remove_node(subgraph_in, center_idx)
            assert subgraph_in.num_nodes - subgraph_out.num_nodes == 1
            score = self.score(in_subgraph=subgraph_in, out_subgraph=subgraph_out, target_idx=node_idx)
            sampling_state.score[sample_idx][i] = score

    def local_subgraph(self, node_mask, center_idx, num_hops):
        '''
        Return the k-hop subgraph induced by a node mask centered at a given node (center node is always included).
        '''
        if torch.is_tensor(center_idx):
            center_idx = center_idx.item()
        node_mask = node_mask.clone() # no mutation
        node_mask[center_idx] = True
        edge_index, _ = subgraph(
            subset=node_mask,
            edge_index=self.graph.edge_index,
            relabel_nodes=False,
        )
        subset, edge_index, mapped_center_idx, _ = k_hop_subgraph(
            node_idx=center_idx,
            num_hops=num_hops,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=self.graph.num_nodes,
        )
        local_subgraph = Data(
            x=self.graph.x[subset],
            edge_index=edge_index,
            y=self.graph.y[subset],
            num_classes=self.graph.num_classes,
        )
        return local_subgraph, mapped_center_idx.item()

class BASE:

    def __init__(self, target_model, graph, loss_fn, config, shadow_models=None, offline=None, offline_threshold_scale_factor=0.7):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config
        self.offline = config.offline if offline is None else offline
        if shadow_models is None:
            self.shadow_models = trainer.train_shadow_models(self.graph, self.loss_fn, self.config)
        else:
            self.shadow_models = shadow_models
        try:
            self.threshold_scale_factor = config.threshold_scale_factor
        except AttributeError:
            self.threshold_scale_factor = offline_threshold_scale_factor if self.offline else 1.0
        print(f'BASE threshold scale factor: {self.threshold_scale_factor}')

    @torch.inference_mode()
    def log_confidence(self, model, x, edge_index, y):
        return F.log_softmax(model(x, edge_index), dim=1)[torch.arange(x.shape[0]), y]

    def run_attack(self, target_node_index):
        x = self.graph.x[target_node_index]
        y = self.graph.y[target_node_index]
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long).to(self.config.device)
        log_conf = self.log_confidence(self.target_model, x, empty_edge_index, y)
        log_conf_shadow = torch.stack([
            self.log_confidence(shadow_model, x, empty_edge_index, y)
            for shadow_model, _ in self.shadow_models
        # Subtracting log(num_shadow_models) is not necessary for attack performance,
        # but should be there if we want to get probabilities by applying sigmoid
        ]).t()
        if self.offline:
            mask = utils.offline_shadow_model_mask(
                target_node_index,
                [train_mask for _, train_mask in self.shadow_models],
            )
            log_conf_shadow = log_conf_shadow[mask].reshape(-1, len(self.shadow_models) // 2)
        threshold = log_conf_shadow.logsumexp(1) - np.log(log_conf_shadow.shape[1])
        score = log_conf - self.threshold_scale_factor * threshold
        return score.sigmoid()

class LaplaceBASE:

    def __init__(self, target_model, graph, loss_fn, config):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config

    def sample_shadow_dataset(self, target_node_index):
        assert target_node_index.shape[0] * 2 <= self.graph.num_nodes
        target_node_mask = index_to_mask(target_node_index, self.graph.num_nodes)
        index_pool = mask_to_index(~target_node_mask)
        perm_index = torch.randperm(index_pool.shape[0])
        index_pool = index_pool[perm_index][:self.graph.num_nodes // 2]
        shadow_train_mask = index_to_mask(index_pool, self.graph.num_nodes)
        shadow_dataset = datasetup.remasked_graph(self.graph, shadow_train_mask)
        assert not torch.any(shadow_train_mask & target_node_mask)
        return shadow_dataset

    def train_shadow_model(self, shadow_dataset):
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.graph.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        shadow_model = utils.fresh_model(
            model_type=config.model,
            num_features=shadow_dataset.num_features,
            hidden_dims=config.hidden_dim,
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
        return shadow_model

    def laplace_approximation(self, shadow_model, shadow_dataset):
        for conv in shadow_model.convs[:-1]:
            for param in conv.parameters():
                param.requires_grad = False
        la = Laplace(
            model=shadow_model,
            likelihood='classification',
            hessian_structure='full',
            subset_of_weights='all',
        )
        def squeeze_collate_fn(batch):
            X, y = batch[0]
            X = X.squeeze(0)
            y = y.squeeze(0)
            return X, y
        train_dataset = datasetup.GraphDatasetWrapper(shadow_dataset)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=squeeze_collate_fn)
        la.fit(train_loader, progress_bar=True)
        la.optimize_prior_precision(pred_type='glm', link_approx='probit')
        return la
    
    def sample_models(self, n_samples, la, shadow_model):
        sampled_weights = la.sample(n_samples)
        sampled_models = []
        for i in range(n_samples):
            sampled_model = copy.deepcopy(shadow_model)
            # Update weights in last layer
            utils.write_flat_params_to_layer(sampled_weights[i], sampled_model.convs[-1])
            sampled_models.append(sampled_model)
        return sampled_models

    def run_attack(self, target_node_index, n_samples=100):
        shadow_dataset = self.sample_shadow_dataset(target_node_index)
        shadow_model = self.train_shadow_model(shadow_dataset)
        la = self.laplace_approximation(shadow_model, shadow_dataset)
        x = self.graph.x[target_node_index]
        y = self.graph.y[target_node_index]
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long).to(self.config.device)
        row_idx = torch.arange(x.shape[0])
        with torch.inference_mode():
            log_conf = F.log_softmax(self.target_model(x, empty_edge_index), dim=1)[row_idx, y]
        if self.config.sample_models:
            log_conf_samples = []
            sampled_models = self.sample_models(n_samples, la, shadow_model)
            with torch.inference_mode():
                for sampled_model in sampled_models:
                    sampled_log_conf = F.log_softmax(sampled_model(x, empty_edge_index), dim=1)[row_idx, y]
                    log_conf_samples.append(sampled_log_conf)
            log_conf_samples = torch.stack(log_conf_samples)
            threshold = log_conf_samples.logsumexp(0) - np.log(len(sampled_models))
        else:
            nfeic = datasetup.NodeFeatureEdgeIndexContainer(x, empty_edge_index)
            la_preds = la(x=nfeic, pred_type='glm', link_approx='probit')
            threshold = la_preds[torch.arange(x.shape[0]), y].log()
        preds = log_conf - threshold
        return preds.sigmoid()

class B_BASE:

    def __init__(self, target_model, graph, loss_fn, config, shadow_models=None):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config
        self.zero_hop_attacker = BASE(
            target_model=target_model,
            graph=graph,
            loss_fn=loss_fn,
            config=config,
            shadow_models=shadow_models,
        )
        self.zero_hop_probs = self.zero_hop_attacker.run_attack(torch.arange(self.graph.num_nodes))
        self.shadow_models = []
        self.train_masks = []
        self.train_shadow_models()

    def train_shadow_models(self):
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.graph.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for _ in tqdm(range(config.additional_shadow_models), desc="Training additional shadow models"):
            train_mask = self.sample_node_mask_zero_hop_MIA()
            shadow_graph = datasetup.remasked_graph(self.graph, train_mask)
            shadow_model = utils.fresh_model(
                model_type=config.model,
                num_features=shadow_graph.num_features,
                hidden_dims=config.hidden_dim,
                num_classes=shadow_graph.num_classes,
                dropout=config.dropout,
            )
            _ = trainer.train_gnn(
                model=shadow_model,
                dataset=shadow_graph,
                config=train_config,
                disable_tqdm=True,
                inductive_split=config.inductive_split,
            )
            shadow_model.eval()
            self.shadow_models.append(shadow_model)
            self.train_masks.append(train_mask)

    def sample_node_mask_zero_hop_MIA(self):
        random_ref = torch.rand(self.graph.num_nodes).to(self.config.device)
        return self.zero_hop_probs > random_ref

    @torch.inference_mode()
    def log_confidence(self, model, x, y):
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long).to(self.config.device)
        return F.log_softmax(model(x, empty_edge_index), dim=1)[torch.arange(x.shape[0]), y]

    def log_model_posterior(self, shadow_models, x, y):
        log_conf = self.log_confidence(self.target_model, x, y)
        threshold = torch.stack([
            self.log_confidence(shadow_model, x, y)
            for shadow_model in shadow_models
        ]).logsumexp(0) - np.log(len(shadow_models))
        return log_conf - threshold

    def run_attack(self, target_node_index):
        preds = torch.zeros_like(target_node_index, dtype=torch.float32)
        for i, target_idx in tqdm(enumerate(target_node_index), total=target_node_index.shape[0], desc="Attacking target nodes using B_BASE"):
            shadow_models_in = []
            shadow_models_out = []
            for shadow_model, train_mask in zip(self.shadow_models, self.train_masks):
                if train_mask[target_idx]:
                    shadow_models_in.append(shadow_model)
                else:
                    shadow_models_out.append(shadow_model)
            if len(shadow_models_in) == 0:
                preds[i] = torch.finfo(preds.dtype).min + 5.0
            elif len(shadow_models_out) == 0:
                preds[i] = torch.finfo(preds.dtype).max - 5.0
            else:
                min_len = min(len(shadow_models_in), len(shadow_models_out))
                np.random.shuffle(shadow_models_in)
                np.random.shuffle(shadow_models_out)
                shadow_models = shadow_models_in[:min_len] + shadow_models_out[:min_len]
                preds[i] = self.log_model_posterior(shadow_models, self.graph.x[target_idx].unsqueeze(0), self.graph.y[target_idx].unsqueeze(0)).squeeze()
        return preds

class BG_BASE:

    def __init__(self, target_model, graph, loss_fn, config, shadow_models=None):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config
        self.zero_hop_attacker = BASE(
            target_model=target_model,
            graph=graph,
            loss_fn=loss_fn,
            config=config,
            shadow_models=shadow_models,
        )
        self.zero_hop_probs = self.zero_hop_attacker.run_attack(torch.arange(self.graph.num_nodes))
        self.shadow_models = []
        self.train_masks = []
        self.train_shadow_models()

    def train_shadow_models(self):
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.graph.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for _ in tqdm(range(config.additional_shadow_models), desc="Training additional shadow models"):
            train_mask = self.sample_node_mask_zero_hop_MIA()
            shadow_graph = datasetup.remasked_graph(self.graph, train_mask)
            shadow_model = utils.fresh_model(
                model_type=config.model,
                num_features=shadow_graph.num_features,
                hidden_dims=config.hidden_dim,
                num_classes=shadow_graph.num_classes,
                dropout=config.dropout,
            )
            _ = trainer.train_gnn(
                model=shadow_model,
                dataset=shadow_graph,
                config=train_config,
                disable_tqdm=True,
                inductive_split=config.inductive_split,
            )
            shadow_model.eval()
            self.shadow_models.append(shadow_model)
            self.train_masks.append(train_mask)

    def sample_node_mask_zero_hop_MIA(self):
        random_ref = torch.rand(self.graph.num_nodes).to(self.config.device)
        return self.zero_hop_probs > random_ref

    @torch.inference_mode()
    def neg_loss(self, model, graph):
        return -F.cross_entropy(model(graph.x, graph.edge_index), graph.y, reduction='sum')

    def loss_signal(self, shadow_models, in_subgraph, out_subgraph):
        target_loss_diff = self.neg_loss(self.target_model, in_subgraph) - self.neg_loss(self.target_model, out_subgraph)
        shadow_loss_diff = torch.tensor([
            self.neg_loss(shadow_model, in_subgraph) - self.neg_loss(shadow_model, out_subgraph)
            for shadow_model in shadow_models
        ]).logsumexp(0) - np.log(len(shadow_models))
        return target_loss_diff - shadow_loss_diff

    def run_attack(self, target_node_index):
        preds = torch.zeros_like(target_node_index, dtype=torch.float32)
        for i, target_idx in tqdm(enumerate(target_node_index), total=target_node_index.shape[0], desc="Attacking target nodes using BG_BASE"):
            shadow_models_in = []
            shadow_models_out = []
            for shadow_model, train_mask in zip(self.shadow_models, self.train_masks):
                if train_mask[target_idx]:
                    shadow_models_in.append(shadow_model)
                else:
                    shadow_models_out.append(shadow_model)
            if len(shadow_models_in) == 0:
                preds[i] = torch.finfo(preds.dtype).min + 5.0
            elif len(shadow_models_out) == 0:
                preds[i] = torch.finfo(preds.dtype).max - 5.0
            else:
                min_len = min(len(shadow_models_in), len(shadow_models_out))
                np.random.shuffle(shadow_models_in)
                np.random.shuffle(shadow_models_out)
                shadow_models = shadow_models_in[:min_len] + shadow_models_out[:min_len]
                loss_sigs = []
                for _ in range(self.config.num_sampled_graphs):
                    node_mask = self.sample_node_mask_zero_hop_MIA()
                    node_mask[target_idx] = True
                    in_subgraph = datasetup.masked_subgraph(self.graph, node_mask)
                    node_mask[target_idx] = False
                    out_subgraph = datasetup.masked_subgraph(self.graph, node_mask)
                    loss_sigs.append(self.loss_signal(shadow_models, in_subgraph, out_subgraph))
                preds[i] = torch.stack(loss_sigs).mean()
        assert preds.shape == target_node_index.shape
        return preds

class S_BASE:

    def __init__(self, target_model, graph, loss_fn, config):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config

    def train_shadow_models(self, target_idx):
        shadow_models = []
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.graph.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        train_mask = self.graph.train_mask.clone()
        for i in range(config.num_shadow_models):
            train_mask[target_idx] = i % 2 == 0
            shadow_dataset = datasetup.remasked_graph(self.graph, train_mask)
            shadow_model = utils.fresh_model(
                model_type=config.model,
                num_features=shadow_dataset.num_features,
                hidden_dims=config.hidden_dim,
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
            shadow_models.append(shadow_model)
        return shadow_models

    @torch.inference_mode()
    def log_confidence(self, model, x, y):
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long).to(self.config.device)
        return F.log_softmax(model(x, empty_edge_index), dim=1)[torch.arange(x.shape[0]), y]

    def log_model_posterior(self, x, y, shadow_models):
        log_conf = self.log_confidence(self.target_model, x, y)
        threshold = torch.stack([
            self.log_confidence(shadow_model, x, y)
            for shadow_model in shadow_models
        # Subtracting log(num_shadow_models) is not necessary for attack performance,
        # but should be there if we want to get probabilities by applying sigmoid
        ]).logsumexp(0) - np.log(self.config.num_shadow_models)
        return log_conf - threshold

    def run_attack(self, target_node_index):
        if self.config.num_processes > 1:
            return self.run_attack_mp(target_node_index)
        preds = torch.zeros_like(target_node_index, dtype=torch.float32)
        for i, target_idx in tqdm(enumerate(target_node_index), total=target_node_index.shape[0], desc="Attacking target nodes"):
            preds[i] = self.compute_pred(target_idx)
        return preds

    def run_attack_mp(self, target_node_index):
        desc = f"Attacking target nodes using S_BASE with {self.config.num_processes} processes"
        with mp.Pool(self.config.num_processes) as pool:
            preds = torch.tensor(pool.map(self.compute_pred, tqdm(target_node_index, desc=desc)))
        assert preds.shape == target_node_index.shape
        return preds

    def compute_pred(self, target_idx):
        shadow_models = self.train_shadow_models(target_idx)
        return self.log_model_posterior(self.graph.x[target_idx].unsqueeze(0), self.graph.y[target_idx].unsqueeze(0), shadow_models).squeeze()

class SG_BASE:

    def __init__(self, target_model, graph, loss_fn, config):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config

    def train_shadow_models(self, target_idx, train_mask):
        config = self.config
        train_mask = train_mask.clone()
        shadow_models = []
        criterion = Accuracy(task="multiclass", num_classes=self.graph.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for i in range(config.num_shadow_models):
            train_mask[target_idx] = i % 2 == 0
            shadow_dataset = datasetup.remasked_graph(self.graph, train_mask)
            shadow_model = utils.fresh_model(
                model_type=config.model,
                num_features=shadow_dataset.num_features,
                hidden_dims=config.hidden_dim,
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
            shadow_models.append(shadow_model)
        return shadow_models

    @torch.inference_mode()
    def neg_loss(self, model, graph):
        return -F.cross_entropy(model(graph.x, graph.edge_index), graph.y, reduction='sum')

    def signal(self, shadow_models, in_subgraph, out_subgraph):
        target_loss_diff = self.neg_loss(self.target_model, in_subgraph) - self.neg_loss(self.target_model, out_subgraph)
        shadow_loss_diff = torch.tensor([
            self.neg_loss(shadow_model, in_subgraph) - self.neg_loss(shadow_model, out_subgraph)
            for shadow_model in shadow_models
        ]).logsumexp(0) - np.log(len(shadow_models))
        return target_loss_diff - shadow_loss_diff

    def run_attack(self, target_node_index):
        if self.config.num_processes > 1:
            return self.run_attack_mp(target_node_index)
        preds = torch.zeros_like(target_node_index, dtype=torch.float32)
        desc = "Attacking target nodes using SG_BASE"
        for i, target_idx in tqdm(enumerate(target_node_index), total=target_node_index.shape[0], desc=desc):
            preds[i] = self.compute_pred(target_idx)
        assert preds.shape == target_node_index.shape
        return preds

    def run_attack_mp(self, target_node_index):
        desc = f"Attacking target nodes using SG_BASE with {self.config.num_processes} processes"
        with mp.Pool(self.config.num_processes) as pool:
            preds = torch.tensor(pool.map(self.compute_pred, tqdm(target_node_index, desc=desc)))
        assert preds.shape == target_node_index.shape
        return preds

    def compute_pred(self, target_idx):
        node_mask = self.graph.train_mask.clone()
        node_mask[target_idx] = True
        in_subgraph = datasetup.masked_subgraph(self.graph, node_mask)
        node_mask[target_idx] = False
        out_subgraph = datasetup.masked_subgraph(self.graph, node_mask)
        shadow_models = self.train_shadow_models(target_idx, node_mask)
        return self.signal(shadow_models, in_subgraph, out_subgraph)

class ConfidenceAttack:

    def __init__(self, target_model, graph, config):
        self.target_model = target_model
        self.graph = graph
        self.config = config

    @torch.inference_mode()
    def run_attack(self, target_node_index):
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long).to(self.config.device)
        return F.softmax(self.target_model(self.graph.x, empty_edge_index), dim=1)[target_node_index, self.graph.y[target_node_index]]

class BMIA:
    '''Preliminary implementation of BMIA. Does not work well.'''

    def __init__(self, target_model, graph, loss_fn, config, eps=1e-11):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config
        self.eps = eps
    
    def sample_shadow_dataset(self, target_node_index):
        assert target_node_index.shape[0] * 2 <= self.graph.num_nodes
        target_node_mask = index_to_mask(target_node_index, self.graph.num_nodes)
        index_pool = mask_to_index(~target_node_mask)
        perm_index = torch.randperm(index_pool.shape[0])
        index_pool = index_pool[perm_index]
        shadow_train_mask = index_to_mask(index_pool, self.graph.num_nodes)
        shadow_dataset = datasetup.remasked_graph(self.graph, shadow_train_mask)
        return shadow_dataset

    def train_shadow_model(self, shadow_dataset):
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.graph.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        shadow_model = utils.fresh_model(
            model_type=config.model,
            num_features=shadow_dataset.num_features,
            hidden_dims=config.hidden_dim,
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
        return shadow_model

    def laplace_approximation(self, shadow_model, shadow_dataset):
        for conv in shadow_model.convs[:-1]:
            for param in conv.parameters():
                param.requires_grad = False
        la = Laplace(
            model=shadow_model,
            likelihood='classification',
            hessian_structure='full',
            subset_of_weights='all',
        )
        def squeeze_collate_fn(batch):
            X, y = batch[0]
            X = X.squeeze(0)
            y = y.squeeze(0)
            return X, y
        train_dataset = datasetup.GraphDatasetWrapper(shadow_dataset)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=squeeze_collate_fn)
        la.fit(train_loader, progress_bar=True)
        la.optimize_prior_precision(pred_type='glm', link_approx='probit', method='marglik')
        return la

    def sample_models(self, n_samples, la, shadow_model):
        sampled_weights = la.sample(n_samples)
        sampled_models = []
        for i in range(n_samples):
            sampled_model = copy.deepcopy(shadow_model)
            # Update weights in last layer
            utils.write_flat_params_to_layer(sampled_weights[i], sampled_model.convs[-1])
            sampled_models.append(sampled_model)
        return sampled_models

    def run_attack(self, target_node_index, n_samples=100):
        config = self.config
        shadow_dataset = self.sample_shadow_dataset(target_node_index)
        shadow_model = self.train_shadow_model(shadow_dataset)
        la = self.laplace_approximation(shadow_model, shadow_dataset)
        x = self.graph.x[target_node_index]
        y = self.graph.y[target_node_index]
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long).to(config.device)
        with torch.inference_mode():
            preds = self.target_model(x, empty_edge_index)
        target_scores = utils.hinge_loss(preds, y)
        sampled_models = self.sample_models(n_samples, la, shadow_model)
        sampled_scores = []
        with torch.inference_mode():
            for sampled_model in sampled_models:
                preds = sampled_model(x, empty_edge_index)
                score = utils.hinge_loss(preds, y)
                sampled_scores.append(score)
        sampled_scores = torch.stack(sampled_scores)
        ds = target_scores - sampled_scores
        d_mean = ds.mean(0)
        d_stdn = ds.std(0) / np.sqrt(n_samples)
        t = d_mean / (d_stdn + self.eps)
        preds = t_dist.cdf(t.cpu().numpy(), n_samples - 1)
        assert preds.shape == target_node_index.shape
        return preds

class LiRA:
    
    EPS = 1e-11

    def __init__(self, target_model, graph, loss_fn, config, shadow_models=None):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config
        self.offline = config.offline
        if shadow_models is None:
            self.shadow_models = trainer.train_shadow_models(self.graph, self.loss_fn, self.config)
        else:
            self.shadow_models = shadow_models

    def query_shadow_models(self, target_node_index, edge_index):
        hinges_in = defaultdict(list)
        hinges_out = defaultdict(list)
        num_target_nodes = target_node_index.shape[0]
        with torch.inference_mode():
            for shadow_model, train_mask in self.shadow_models:
                preds = shadow_model(self.graph.x[target_node_index], edge_index)
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
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long).to(self.config.device)
        mean_in, std_in, mean_out, std_out = self.query_shadow_models(target_node_index, empty_edge_index)
        with torch.inference_mode():
            preds = self.target_model(self.graph.x[target_node_index], empty_edge_index)
            target_hinges = utils.hinge_loss(preds, self.graph.y[target_node_index])
        if self.offline:
            # In offline LiRA the test statistic is Lambda = 1 - P(Z > conf_target), where Z is a sample from
            # a normal distribution with mean and variance given by the shadow models confidences.
            # We normalize the target confidence and compute the test statistic Lambda' = P(Z < x), Z ~ Normal(0, 1)
            # For numerical stability, compute the log CDF.
            score = norm.logcdf(
                target_hinges.cpu().numpy(),
                loc=mean_out.cpu().numpy(),
                scale=std_out.cpu().numpy() + self.EPS,
            )
            score = torch.tensor(score)
        else:
            p_in = norm.logpdf(
                target_hinges.cpu().numpy(),
                loc=mean_in.cpu().numpy(),
                scale=std_in.cpu().numpy() + self.EPS,
            )
            p_out = norm.logpdf(
                target_hinges.cpu().numpy(),
                loc=mean_out.cpu().numpy(),
                scale=std_out.cpu().numpy() + self.EPS,
            )
            assert p_in.shape == p_out.shape == (target_node_index.shape[0],)
            score = torch.tensor(p_in - p_out)
        return score

class RMIA:

    def __init__(self, target_model, graph, loss_fn, config, shadow_models=None):
        self.target_model = target_model
        self.graph = graph
        self.loss_fn = loss_fn
        self.config = config
        self.offline = config.offline
        self.num_z = int(config.Z_frac * graph.num_nodes)
        self.z_set = torch.randperm(self.graph.num_nodes)[:self.num_z].sort()[0]
        if shadow_models is None:
            self.shadow_models = trainer.train_shadow_models(self.graph, self.loss_fn, self.config)
        else:
            self.shadow_models = shadow_models
        if self.offline:
            try:
                self.interp_param = config.interp_param
            except AttributeError:
                self.interp_param = None
        else:
            self.interp_param = None

    @torch.inference_mode()
    def run_attack(self, target_node_index):
        empty_edge_index = torch.tensor([[],[]], dtype=torch.long).to(self.config.device)
        num_target_nodes = target_node_index.shape[0]
        # Compute ratio_x over target nodes
        p_x = []
        for shadow_model, _ in self.shadow_models:
            preds = F.softmax(
                shadow_model(self.graph.x[target_node_index], empty_edge_index),
                dim=1
            )[torch.arange(target_node_index.shape[0]), self.graph.y[target_node_index]]
            p_x.append(preds)
        p_x = torch.stack(p_x).t()
        assert p_x.shape == (num_target_nodes, len(self.shadow_models))
        if self.offline:
            mask = utils.offline_shadow_model_mask(target_node_index, [train_mask for _, train_mask in self.shadow_models])
            p_x_out = p_x[mask].reshape(-1, len(self.shadow_models) // 2)
            p_x_out = p_x_out.mean(dim=1)
            p_x = 0.5 * ((1 + self.interp_param) * p_x_out + 1 - self.interp_param)
        else:
            p_x = p_x.mean(dim=1)
        p_x_target = F.softmax(
            self.target_model(self.graph.x[target_node_index], empty_edge_index),
            dim=1,
        )[torch.arange(target_node_index.shape[0]), self.graph.y[target_node_index]]
        ratio_x = p_x_target / p_x
        # Compute ratio_z over self.z_set
        p_z = []
        for shadow_model, _ in self.shadow_models:
            preds = F.softmax(
                shadow_model(self.graph.x[self.z_set], empty_edge_index),
                dim=1
            )[torch.arange(self.num_z), self.graph.y[self.z_set]]
            p_z.append(preds)
        p_z = torch.stack(p_z)
        assert p_z.shape == (len(self.shadow_models), self.num_z)
        p_z_target = F.softmax(
            self.target_model(self.graph.x[self.z_set], empty_edge_index),
            dim=1,
        )[torch.arange(self.num_z), self.graph.y[self.z_set]]
        ratio_z = p_z_target / p_z
        score = torch.tensor([(x > ratio_z * self.config.rmia_gamma).float().mean().item() for x in ratio_x])
        return score
