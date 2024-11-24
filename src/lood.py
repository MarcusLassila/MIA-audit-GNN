import datasetup
import trainer
import utils

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import subgraph
from torchmetrics import Accuracy

class LOOD:
    
    def __init__(self, config, num_inference_samples=10000):
        self.config = config
        self.num_inference_samples = num_inference_samples

    def train_model(self, dataset):
        config = self.config
        model = utils.fresh_model(
            model_type=self.config.model,
            num_features=dataset.num_features,
            hidden_dims=config.hidden_dim_target * 2,
            num_classes=dataset.num_classes,
            dropout=config.dropout,
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
            inductive_split=config.inductive_split,
        )
        model.eval()
        return model

    def measure(self, dataset, query):
        incl_model = self.train_model(dataset)
        incl_model.dropout_during_inference = True
        preds = []
        with torch.no_grad():
            for _ in range(10000):
                preds.append(incl_model(dataset.x, dataset.edge_index)[0, dataset.y[0]])
        preds = torch.tensor(preds)
        mean = preds.mean()
        std = preds.std()
        utils.plot_histogram_and_fitted_gaussian(preds, mean, std, bins=50)
        # train_set = datasetup.masked_subgraph(dataset, dataset.train_mask)
        # test_set = datasetup.masked_subgraph(dataset, dataset.test_mask)
        # nodes = torch.arange(train_set.num_nodes)
        # for node in range(train_set.num_nodes):
        #     incl = torch.cat((nodes[:node], nodes[node + 1:]))
        #     sub_edge_index, _ = subgraph(
        #         subset=train_set.x[incl],
        #         edge_index=train_set.edge_index,
        #         relabel_nodes=True,
        #         num_nodes=train_set.num_nodes,
        #     )
        #     sub_graph = datasetup.merge_graphs(train_set, test_set)
        #     excl_model = self.train_model(sub_graph)
        #     for _ in range(1):
        #         excl_pred = excl_model()