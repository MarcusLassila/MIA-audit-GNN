import datasetup
import evaluation
import models
import trainer
import utils

import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm.auto import tqdm


class BasicShadowAttack:
    
    def __init__(self, target_model, shadow_dataset, config):
        target_model.eval()
        self.target_model = target_model
        self.shadow_model = utils.fresh_model(
            model_type=config.model,
            num_features=shadow_dataset.num_features,
            hidden_dim=config.hidden_dim_target,
            num_classes=shadow_dataset.num_classes,
            dropout=config.dropout,
        )
        self.attack_model = models.MLP(in_dim=shadow_dataset.num_classes, hidden_dims=config.hidden_dim_attack)
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
            optimizer=getattr(torch.optim, config.optimizer),
        )
        train_res = trainer.train_gnn(
            model=self.shadow_model,
            dataset=self.shadow_dataset,
            config=train_config,
        )
        evaluation.evaluate_graph_training(
            model=self.shadow_model,
            dataset=self.shadow_dataset,
            criterion=train_config.criterion,
            training_results=train_res if self.plot_training_results else None,
            plot_title="Shadow model",
            savedir=config.savedir,
        )

    def train_attack_model(self):
        config = self.config
        train_dataset, valid_dataset = datasetup.create_attack_dataset(self.shadow_dataset, self.shadow_model)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
        train_config = trainer.TrainConfig(
            criterion=Accuracy(task="multiclass", num_classes=2).to(config.device),
            device=config.device,
            epochs=config.epochs_attack,
            early_stopping=config.early_stopping,
            loss_fn=nn.CrossEntropyLoss(),
            lr=1e-3,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        trainer.train_mlp(
            model=self.attack_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            config=train_config,
        )
    
    def run_attack(self, target_samples):
        config = self.config
        num_target_samples = target_samples.x.shape[0]
        with torch.inference_mode():
            preds = evaluation.k_hop_query(
                model=self.target_model,
                dataset=target_samples,
                query_nodes=[*range(num_target_samples)],
                num_hops=config.query_hops,
            )
            logits = self.attack_model(preds)[:,1]
            labels = target_samples.train_mask.long()
        return evaluation.bc_evaluation(logits, labels)

class ConfidenceAttack:
    
    def __init__(self, target_model, config):
        target_model.eval()
        self.target_model = target_model
        self.config = config
    
    def run_attack(self, target_samples):
        config = self.config
        num_target_samples = target_samples.x.shape[0]
        with torch.inference_mode():
            preds = evaluation.k_hop_query(
                model=self.target_model,
                dataset=target_samples,
                query_nodes=[*range(num_target_samples)],
                num_hops=config.query_hops,
            )
            row_idx = np.arange(num_target_samples)
            confidences = F.softmax(preds, dim=1)[row_idx, target_samples.y]
            labels = target_samples.train_mask.long()
        return evaluation.bc_evaluation(confidences, labels)

class LiRA:
    '''
    The likelihood ratio attack from "Membership Inference Attacks From First Principles"
    '''
    EPS = 1e-6

    def __init__(self, target_model, population, config, online=False):
        target_model.eval()
        self.target_model = target_model
        self.shadow_models = []
        self.population = population # Should not contain target samples.
        self.config = config
        self.shadow_size = population.x.shape[0] // 2
        self.train_shadow_models()

    def train_shadow_models(self):
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.population.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=F.cross_entropy,
            lr=config.lr,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for _ in tqdm(range(config.num_shadow_models), desc=f"Training {config.num_shadow_models} shadow models for LiRA"):
            shadow_dataset = datasetup.sample_subgraph(self.population, self.shadow_size)
            shadow_model = utils.fresh_model(
                model_type=config.model,
                num_features=shadow_dataset.num_features,
                hidden_dim=config.hidden_dim_target,
                num_classes=shadow_dataset.num_classes,
                dropout=config.dropout,
            )
            _ = trainer.train_gnn(
                model=shadow_model,
                dataset=shadow_dataset,
                config=train_config,
                use_tqdm=False,
            )
            self.shadow_models.append(shadow_model)
    
    def get_mean_and_std(self, target_samples):
        config = self.config
        logits = []
        num_target_samples = target_samples.x.shape[0]
        row_idx = np.arange(num_target_samples)
        desc=f"Computing confidence values from shadow models. {next(self.shadow_models[0].parameters()).device} device"
        for shadow_model in tqdm(self.shadow_models, desc=desc):
            shadow_model.eval()
            with torch.inference_mode():
                preds = evaluation.k_hop_query(
                    model=shadow_model,
                    dataset=target_samples,
                    query_nodes=[*range(num_target_samples)],
                    num_hops=config.query_hops,
                )
                confs = F.softmax(preds, dim=1)[row_idx, target_samples.y]
                logits.append(confs.logit(eps=LiRA.EPS)) # Logit scaling for approximately normal distribution.
        logits = torch.stack(logits)
        assert logits.shape == torch.Size([len(self.shadow_models), num_target_samples])
        means = logits.mean(dim=0)
        stds = logits.std(dim=0)
        if config.experiments == 1:
            utils.plot_histogram_and_fitted_gaussian(
                x=logits[:,0].cpu().numpy(),
                mean=means[0].cpu().numpy(),
                std=stds[0].cpu().numpy(),
                bins=max(len(self.shadow_models) // 8, 1),
                savepath="./results/gaussian_fit_histogram.png",
            )
        return means, stds

    def run_attack(self, target_samples):
        config = self.config
        target_samples.to(config.device)
        num_target_samples = target_samples.x.shape[0]
        means, stds = self.get_mean_and_std(target_samples)
        with torch.inference_mode():
            preds = evaluation.k_hop_query(
                model=self.target_model,
                dataset=target_samples,
                query_nodes=[*range(num_target_samples)],
                num_hops=config.query_hops,
            )
            row_idx = np.arange(num_target_samples)
            target_logits = F.softmax(preds, dim=1)[row_idx, target_samples.y].logit(eps=LiRA.EPS)

        # In offline LiRA the test statistic is Lambda = 1 - P(Z > conf_target), where Z is a sample from
        # a normal distribution with mean and variance given by the shadow models confidences.
        # We normalize the target confidence and compute the test statistic Lambda' = P(Z < x), Z ~ Normal(0, 1)
        # For numerical stability, compute the log CDF.
        preds = norm.logcdf(
            target_logits.cpu().numpy(),
            loc=means.cpu().numpy(),
            scale=stds.cpu().numpy()
        )
        truth = target_samples.train_mask.long().cpu().numpy()
        return evaluation.bc_evaluation(
            preds=preds,
            labels=truth,
        )

class RMIA:
    '''
    The attack from "Low-Cost High-Power Membership Inference Attacks".
    Only the offline attack is currently supported.
    '''

    def __init__(self, target_model, population, config, online=False):
        target_model.eval()
        self.target_model = target_model
        self.population = population
        self.config = config
        self.online = online
        self.out_models = []
        self.out_size = self.population .x.shape[0] // 2
        self.gamma = 2 # Value used in the original paper
        self.a = config.rmia_offline_interp_param
        self.train_out_models()

    def train_out_models(self):
        config = self.config
        criterion = Accuracy(task="multiclass", num_classes=self.population.num_classes).to(config.device)
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=F.cross_entropy,
            lr=config.lr,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for _ in tqdm(range(config.num_shadow_models), desc=f"Training {config.num_shadow_models} out models for RMIA"):
            shadow_dataset = datasetup.sample_subgraph(self.population, self.out_size)
            shadow_model = utils.fresh_model(
                model_type=config.model,
                num_features=shadow_dataset.num_features,
                hidden_dim=config.hidden_dim_target,
                num_classes=shadow_dataset.num_classes,
                dropout=config.dropout,
            )
            _ = trainer.train_gnn(
                model=shadow_model,
                dataset=shadow_dataset,
                config=train_config,
                use_tqdm=False,
            )
            self.out_models.append(shadow_model)

    def ratio(self, dataset):
        config = self.config
        num_target_nodes = dataset.x.shape[0]
        row_idx = np.arange(num_target_nodes)
        out_confidences = []
        for shadow_model in tqdm(self.out_models, desc=f"Querying out models. {next(self.out_models[0].parameters()).device} device"):
            shadow_model.eval()
            with torch.inference_mode():
                preds = evaluation.k_hop_query(
                    model=shadow_model,
                    dataset=dataset,
                    query_nodes=[*range(num_target_nodes)],
                    num_hops=config.query_hops,
                )
                out_confidences.append(F.softmax(preds, dim=1)[row_idx, dataset.y])
        out_confidences = torch.stack(out_confidences)
        pr_out = out_confidences.mean(dim=0)
        pr = 0.5 * ((self.a + 1) * pr_out + 1 - self.a) # Heuristic to approximate the average of in and out, from out only.

        with torch.inference_mode():
            preds = evaluation.k_hop_query(
                model=self.target_model,
                dataset=dataset,
                query_nodes=[*range(num_target_nodes)],
                num_hops=config.query_hops,
            )
            target_confidence = F.softmax(preds, dim=1)[row_idx, dataset.y] 
        assert pr.shape == target_confidence.shape == torch.Size([num_target_nodes])
        return target_confidence / pr

    def score(self, target_samples):
        ratioX = self.ratio(target_samples)
        ratioZ = self.ratio(self.population)
        thresholds = ratioZ * self.gamma
        count = torch.zeros_like(ratioX)
        for i, x in enumerate(ratioX):
            count[i] = (x > thresholds).sum().item()
        sizeZ = thresholds.shape[0]
        return count / sizeZ
    
    def run_attack(self, target_samples):
        target_samples.to(self.config.device)
        self.population.to(self.config.device)
        preds = self.score(target_samples)
        labels = target_samples.train_mask.long()
        return evaluation.bc_evaluation(preds=preds, labels=labels) # Beta parameter is sweeped when computing roc/auroc.
