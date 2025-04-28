import sys
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_dir, '..', 'src'))

from src import attacks, datasetup, evaluation, trainer, utils
import unittest
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import numpy as np
from sklearn.metrics import roc_curve
import yaml

class TestEvaluation(unittest.TestCase):

    def test_fpr_threshold(self):
        with open("default_parameters.yaml", "r") as file:
            config = utils.Config(yaml.safe_load(file)['default-parameters'])
            config.device = 'cpu'
        dataset = datasetup.parse_dataset(config.datadir, 'cora')
        _ = datasetup.random_remasked_graph(dataset, train_frac=0.5, val_frac=0.0, mutate=True)
        criterion = Accuracy(task="multiclass", num_classes=dataset.num_classes)
        target_model = utils.fresh_model(
            model_type=config.model,
            num_features=dataset.num_features,
            hidden_dims=config.hidden_dim,
            num_classes=dataset.num_classes,
            dropout=config.dropout,
        )
        train_config = trainer.TrainConfig(
            criterion=criterion,
            device=config.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=F.cross_entropy,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        _ = trainer.train_gnn(
            model=target_model,
            dataset=dataset,
            config=train_config,
            inductive_split=config.inductive_split,
        )
        attacker = attacks.ConfidenceAttack(
            target_model=target_model,
            graph=dataset,
            config=config,
        )
        target_node_index = torch.randperm(dataset.num_nodes)[:500]
        preds = attacker.run_attack(target_node_index=target_node_index).numpy()
        truth = dataset.train_mask.long().numpy()[target_node_index]
        fpr, tpr, thresholds = roc_curve(y_true=truth, y_score=preds)
        tpr_fixed_fpr, threshold = utils.tpr_at_fixed_fpr(fpr, tpr, config.target_fpr, thresholds)
        hard_preds = (preds >= threshold).astype(np.int64)
        tp = (hard_preds & truth).sum()
        fp = (hard_preds & (truth ^ 1)).sum()
        fn = ((hard_preds ^ 1) & truth).sum()
        tn = ((hard_preds ^ 1) & (truth ^ 1)).sum()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        self.assertGreaterEqual(config.target_fpr, fpr)
        self.assertGreaterEqual(tpr, tpr_fixed_fpr)

if __name__ == '__main__':
    unittest.main()
