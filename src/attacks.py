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
    
    def __init__(self, dataset, target_model, shadow_model, config):
        # TODO: Allow for pretrained target model.
        self.target_dataset, self.shadow_dataset = datasetup.target_shadow_split(dataset, config.split)
        self.target_model = target_model
        self.shadow_model = shadow_model
        self.attack_model = models.MLP(in_dim=dataset.num_classes, hidden_dims=config.hidden_dim_attack)
        self.criterion_target = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(config.device)
        self.config = config
        self.plot_training_results = True
    
    def train_all(self):
        config = self.config
        train_config = trainer.TrainConfig(
            criterion=self.criterion_target,
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=F.nll_loss,
            lr=config.lr,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        self.train_graph_model(
            model=self.target_model,
            dataset=self.target_dataset,
            config=train_config,
            name='Target',
        )
        self.train_graph_model(
            model=self.shadow_model,
            dataset=self.shadow_dataset,
            config=train_config,
            name='Shadow',
        )
        self.train_attack_model()

    def train_graph_model(self, model, dataset, config: trainer.TrainConfig, name='unnamed'):
        train_res = trainer.train_gnn(
            model=model,
            dataset=dataset,
            config=config,
        )
        if self.plot_training_results:
            utils.plot_training_results(train_res, name, self.config.savedir)
        test_score = evaluation.evaluate_graph_model(
            model=model,
            dataset=dataset,
            mask=dataset.test_mask,
            criterion=config.criterion,
        )
        print(f"Test accuracy: {test_score:.4f}")
        
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
        self.train_all()
        target_scores = {
            'train_score': evaluation.evaluate_graph_model(
                model=self.target_model,
                dataset=self.target_dataset,
                mask=self.target_dataset.train_mask,
                criterion=self.criterion_target,
            ),
            'test_score': evaluation.evaluate_graph_model(
                model=self.target_model,
                dataset=self.target_dataset,
                mask=self.target_dataset.test_mask,
                criterion=self.criterion_target,
            ),
        }
        eval_metrics = evaluation.evaluate_shadow_attack(self.attack_model, self.target_model, self.target_dataset, num_hops=config.query_hops)
        return dict(eval_metrics, **target_scores)
    

class ConfidenceAttack:
    
    def __init__(self, dataset, target_model, config):
        self.dataset = dataset
        self.target_model = target_model
        self.config = config
        self.criterion = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(config.device)
        self.plot_training_results = True
        self.is_pretrained = False

    def train_target(self):
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
            model=self.target_model,
            dataset=self.dataset,
            config=train_config,
        )
        if self.plot_training_results:
            utils.plot_training_results(train_res, 'Target', self.config.savedir)
        test_score = evaluation.evaluate_graph_model(
            model=self.target_model,
            dataset=self.dataset,
            mask=self.dataset.test_mask,
            criterion=self.criterion,
        )
        print(f"Test accuracy: {test_score:.4f}")
    
    def run_attack(self):
        config = self.config
        if not self.is_pretrained:
            self.train_target()
        target_scores = {
            'train_score': evaluation.evaluate_graph_model(
                model=self.target_model,
                dataset=self.dataset,
                mask=self.dataset.train_mask,
                criterion=self.criterion
            ),
            'test_score': evaluation.evaluate_graph_model(
                model=self.target_model,
                dataset=self.dataset,
                mask=self.dataset.test_mask,
                criterion=self.criterion
            ),
        }
        eval_metrics = evaluation.evaluate_confidence_attack(
            target_model=self.target_model,
            dataset=self.dataset,
            threshold=config.confidence_threshold,
            num_hops=config.query_hops,
        )
        return dict(target_scores, **eval_metrics)
