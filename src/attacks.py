import datasetup
import evaluation
import models
import trainer
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy


class BasicShadowAttack:
    
    def __init__(self, target_model, shadow_dataset, target_dataset, config):
        self.target_model = target_model
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
        target_scores = {
            'train_score': evaluation.evaluate_graph_model(
                model=self.target_model,
                dataset=self.target_dataset,
                mask=self.target_dataset.train_mask,
                criterion=self.criterion,
            ),
            'test_score': evaluation.evaluate_graph_model(
                model=self.target_model,
                dataset=self.target_dataset,
                mask=self.target_dataset.test_mask,
                criterion=self.criterion,
            ),
        }
        eval_metrics = evaluation.evaluate_shadow_attack(
            attack_model=self.attack_model,
            target_model=self.target_model,
            dataset=self.target_dataset,
            num_hops=config.query_hops
        )
        return dict(eval_metrics, **target_scores)
    

class ConfidenceAttack:
    
    def __init__(self, target_model, target_dataset, config):
        self.target_model = target_model
        self.target_dataset = target_dataset
        self.config = config
        self.criterion = Accuracy(task="multiclass", num_classes=target_dataset.num_classes).to(config.device)
        self.plot_training_results = True
        self.is_pretrained = False
    
    def run_attack(self):
        config = self.config
        target_scores = {
            'train_score': evaluation.evaluate_graph_model(
                model=self.target_model,
                dataset=self.target_dataset,
                mask=self.target_dataset.train_mask,
                criterion=self.criterion
            ),
            'test_score': evaluation.evaluate_graph_model(
                model=self.target_model,
                dataset=self.target_dataset,
                mask=self.target_dataset.test_mask,
                criterion=self.criterion
            ),
        }
        eval_metrics = evaluation.evaluate_confidence_attack(
            target_model=self.target_model,
            dataset=self.target_dataset,
            threshold=config.confidence_threshold,
            num_hops=config.query_hops,
        )
        return dict(target_scores, **eval_metrics)
