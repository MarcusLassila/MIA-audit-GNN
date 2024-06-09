import torch
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

def create_attack_dataset(shadow_dataset, shadow_model):
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
        num_classes=dataset.num_classes,
        num_features=dataset.num_features,
        name=dataset.name,
    )

def sample_subgraph(dataset, num_nodes, train_frac=0.5, val_frac=0.2, keep_class_proportions=True):
    total_num_nodes = dataset.x.shape[0]
    assert 0 < num_nodes <= total_num_nodes
    node_frac = num_nodes / total_num_nodes
    randomized_index = torch.randperm(total_num_nodes)
    if keep_class_proportions:
        node_index = []
        for c in range(dataset.num_classes):
            idx = (dataset.y[randomized_index] == c).nonzero().squeeze(dim=1)
            n = int(len(idx) * node_frac)
            node_index.append(idx[:n])
        node_index = torch.cat(node_index)
    else:
        node_index = randomized_index[:num_nodes]
    return extract_subgraph(dataset, node_index, train_frac=train_frac, val_frac=val_frac)

def disjoint_split(dataset, balance=0.5):
    node_index_A = []
    node_index_B = []
    num_nodes = dataset.x.shape[0]
    randomized_index = torch.randperm(num_nodes)
    for c in range(dataset.num_classes):
        idx = (dataset.y[randomized_index] == c).nonzero().squeeze(dim=1)
        n = int(len(idx) * balance)
        node_index_A.append(idx[:n])
        node_index_B.append(idx[n:])
    node_index_A = torch.cat(node_index_A)
    node_index_B = torch.cat(node_index_B)
    return node_index_A, node_index_B

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
        target_index, shadow_index = disjoint_split(dataset, balance=target_frac)
        target_set = extract_subgraph(dataset, target_index)
        shadow_set = extract_subgraph(dataset, shadow_index)
    else:
        raise ValueError(f"Unsupported split: {split}")
    return target_set, shadow_set
