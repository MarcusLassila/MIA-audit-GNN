from evaluation import query_attack_features

import torch
import torch_geometric.datasets as datasets
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask, subgraph
from sklearn.model_selection import train_test_split


class AttackDataset(torch.utils.data.Dataset):
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

def create_attack_dataset(shadow_dataset, shadow_model, k_hop_queries=False, num_hops=0):
    if k_hop_queries:
        features = query_attack_features(shadow_model, shadow_dataset, range(shadow_dataset.x.shape[0]), num_hops=num_hops).cpu()
    else:
        features = shadow_model(shadow_dataset.x, shadow_dataset.edge_index).cpu()
    labels = shadow_dataset.train_mask.long().cpu()
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, stratify=labels)
    train_dataset = AttackDataset(train_X, train_y)
    test_dataset = AttackDataset(test_X, test_y)
    return train_dataset, test_dataset

def extract_subgraph(dataset, node_index, train_frac=0.5, val_frac=0.2):
    edge_index, _ = subgraph(
        subset=node_index,
        edge_index=dataset.edge_index,
        relabel_nodes=True,
        num_nodes=dataset.x.shape[0],
    )
    num_nodes = len(node_index)
    num_train_nodes = int(train_frac * num_nodes)
    num_val_nodes = int(val_frac * num_nodes)
    train_index = torch.arange(0, num_train_nodes)
    val_index = torch.arange(num_train_nodes, num_train_nodes + num_val_nodes)
    test_index = torch.arange(num_train_nodes + num_val_nodes, num_nodes)
    train_mask = index_to_mask(train_index, num_nodes)
    val_mask = index_to_mask(val_index, num_nodes)
    test_mask = index_to_mask(test_index, num_nodes)
    return Data(
        x=dataset.x[node_index],
        edge_index=edge_index,
        y=dataset.y[node_index],
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=dataset.num_classes, # The number of classes is the same as for the overall dataset, even if some class would not be represented in the sample.
        num_features=dataset.num_features,
        name=dataset.name,
    )

def sample_subgraph(dataset, num_nodes, train_frac=0.5, val_frac=0.2):
    total_num_nodes = dataset.x.shape[0]
    assert 0 < num_nodes <= total_num_nodes
    node_index = torch.randperm(total_num_nodes)[:num_nodes]
    edge_index, _ = subgraph(
        subset=node_index,
        edge_index=dataset.edge_index,
        relabel_nodes=True,
        num_nodes=total_num_nodes,
    )
    num_train_nodes = int(train_frac * num_nodes)
    num_val_nodes = int(val_frac * num_nodes)
    train_index = torch.arange(0, num_train_nodes)
    val_index = torch.arange(num_train_nodes, num_train_nodes + num_val_nodes)
    test_index = torch.arange(num_train_nodes + num_val_nodes, num_nodes)
    train_mask = index_to_mask(train_index, num_nodes)
    val_mask = index_to_mask(val_index, num_nodes)
    test_mask = index_to_mask(test_index, num_nodes)
    return Data(
        x=dataset.x[node_index],
        edge_index=edge_index,
        y=dataset.y[node_index],
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=dataset.num_classes, # The number of classes is the same as for the overall dataset, even if some class would not be represented in the sample.
        num_features=dataset.num_features,
        name=dataset.name,
    )

def target_shadow_split(dataset, split="sampled", target_frac=0.5, shadow_frac=0.5):
    num_nodes = dataset.x.shape[0]
    if split == "sampled":
        assert 0.0 < target_frac <= 1.0 and 0.0 <= shadow_frac <= 1.0
        target_size = int(num_nodes * target_frac)
        shadow_size = int(num_nodes * shadow_frac)
        target_set = sample_subgraph(dataset, target_size)
        shadow_set = sample_subgraph(dataset, shadow_size)
    elif split == "disjoint":
        assert 0.0 < target_frac + shadow_frac <= 1.0
        target_size = int(num_nodes * target_frac)
        shadow_size = int(num_nodes * shadow_frac)
        node_index = torch.randperm(num_nodes)
        target_index = node_index[:target_size]
        shadow_index = node_index[target_size: target_size + shadow_size]
        target_set = extract_subgraph(dataset, target_index)
        shadow_set = extract_subgraph(dataset, shadow_index)
    elif split == "TSTF":
        target_set, shadow_set = target_shadow_split(dataset, split="sampled", target_frac=1.0, shadow_frac=1.0)
    return target_set, shadow_set
