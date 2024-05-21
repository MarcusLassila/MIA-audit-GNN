import datasetup
import evaluation
import models
import trainer
import utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm.auto import tqdm


class BasicShadowAttack:
    
    def __init__(self, target_model, shadow_dataset, target_dataset, config):
        self.target_model = target_model
        self.target_model.eval()
        self.shadow_model = utils.fresh_model(
            model_type=config.model,
            num_features=shadow_dataset.num_features,
            hidden_dim=config.hidden_dim_target,
            num_classes=shadow_dataset.num_classes,
            dropout=config.dropout
        )
        self.attack_model = models.MLP(in_dim=shadow_dataset.num_classes, hidden_dims=config.hidden_dim_attack)
        self.shadow_dataset = shadow_dataset
        self.target_dataset = target_dataset
        self.criterion = Accuracy(task="multiclass", num_classes=shadow_dataset.num_classes).to(config.device)
        self.config = config
        self.plot_training_results = True
    
    def train_shadow_model(self):
        config = self.config
        train_config = trainer.TrainConfig(
            criterion=self.criterion,
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=F.nll_loss,
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
            savedir=config.savedir
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
    
    def run_attack(self):
        config = self.config
        self.train_shadow_model()
        self.train_attack_model()
        return evaluation.evaluate_shadow_attack(
            attack_model=self.attack_model,
            target_model=self.target_model,
            dataset=self.target_dataset,
            num_hops=config.query_hops
        )

class ConfidenceAttack:
    
    def __init__(self, target_model, target_dataset, config):
        self.target_model = target_model
        self.target_model.eval()
        self.target_dataset = target_dataset
        self.config = config
        self.criterion = Accuracy(task="multiclass", num_classes=target_dataset.num_classes).to(config.device)
        self.plot_training_results = True
        self.is_pretrained = False
    
    def run_attack(self):
        config = self.config
        return evaluation.evaluate_confidence_attack(
            target_model=self.target_model,
            dataset=self.target_dataset,
            threshold=config.confidence_threshold,
            num_hops=config.query_hops,
        )

class OfflineLiRA:

    def __init__(self, target_model, population, config):
        self.target_model = target_model
        self.target_model.eval()
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
            loss_fn=F.nll_loss,
            lr=config.lr,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        for _ in tqdm(range(config.num_shadow_models), desc=f"Training {config.num_shadow_models} shadow models for LiRA."):
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
        confidences = []
        for shadow_model in self.shadow_models:
            shadow_model.eval()
            with torch.inference_mode():
                features = evaluation.query_attack_features(
                    model=shadow_model,
                    dataset=target_samples,
                    query_nodes=range(target_samples.x.shape[0]),
                    num_hops=config.query_hops
                )
                confidences.append(F.softmax(features, dim=1).max(dim=1).values)
        confidences = torch.stack(confidences)
        assert confidences.shape == torch.Size([len(self.shadow_models), target_samples.x.shape[0]])
        means = confidences.mean(dim=0)
        stds = confidences.std(dim=0)
        return means, stds
    
    def run_attack(self, target_samples):
        config = self.config
        means, stds = self.get_mean_and_std(target_samples)
        with torch.inference_mode():
            features = evaluation.query_attack_features(
                model=self.target_model,
                dataset=target_samples,
                query_nodes=range(target_samples.x.shape[0]),
                num_hops=config.query_hops,
            )
            target_confidences = F.softmax(features, dim=1).max(dim=1).values

        # In offline LiRA the test statistic is Lambda = 1 - P(Z > conf_target), where Z is a sample from
        # a normal distribution with mean and variance given by the shadow models confidences.
        # We normalize the target confidence and compute the test statistic Lambda' = 1 - P(Z > x), Z ~ Normal(0, 1).
        # We then use 1 - P(Z > x) = 0.5[1 + erf(x / sqrt(2))], Z ~ Normal(0, 1).
        target_confidences_normalized = (target_confidences - means) / (stds + 1e-8)
        pred_proba = 0.5 * (1 + torch.erf(target_confidences_normalized / np.sqrt(2)))
        truth = target_samples.train_mask.long()
        return evaluation.bc_evaluation(
            preds=pred_proba,
            labels=truth,
            device=config.device,
            threshold=0.5
        )
