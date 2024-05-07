import torch
import torch_geometric
import torch_geometric.datasets as datasets
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

torch.manual_seed(1)

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
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=50, stratify=labels) # test_size=0.2
    train_dataset = AttackDataset(train_X, train_y)
    test_dataset = AttackDataset(test_X, test_y)
    return train_dataset, test_dataset

def index_to_mask(size, index):
    mask = torch.zeros(size, dtype=bool)
    mask[index] = True
    return mask

def internal_edges(dataset, node_index):
    index_set = set(node_index.numpy())
    edge_mask = torch.tensor([x.item() in index_set and y.item() in index_set for x, y in dataset.edge_index.t()])
    return dataset.edge_index[:,edge_mask]

def extract_subgraph(dataset, node_index, train_frac=0.2, val_frac=0.2):
    '''
    Args:
        dataset: dataset of the full graph.
        node_index: index of a subset of the full graph.
        train_frac: fraction node_index to be included in train_mask
        val_frac: fraction of node_index to be included in val_mask
    
    test_mask include the remaining of node_index.
    
    Return:
        Data object where only edges between node_index nodes remains in edge_index,
        and train_mask, val_mask and test_mask only include nodes in node_index.
        
    Note that there is no guarantee that the train/val/test split have any balanced
    representation of class labels or similarly connected as the full graph.
    '''
    total_num_nodes = dataset[0].num_nodes
    edge_index = internal_edges(dataset, node_index)
    num_nodes = len(node_index)
    num_train_nodes = int(train_frac * num_nodes)
    num_val_nodes = int(val_frac * num_nodes)
    train_index = node_index[:num_train_nodes]
    val_index = node_index[num_train_nodes: num_train_nodes + num_val_nodes]
    test_index = node_index[num_train_nodes + num_val_nodes:]
    train_mask = index_to_mask(total_num_nodes, train_index)
    val_mask = index_to_mask(total_num_nodes, val_index)
    test_mask = index_to_mask(total_num_nodes, test_index)
    return Data(
        x=dataset.x,
        edge_index=edge_index,
        y=dataset.y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=dataset.num_classes,
        num_features=dataset.num_features,
        name=dataset.name,
    )

def sample_subgraph(dataset, num_nodes, train_frac=0.2, val_frac=0.2):
    total_num_nodes = dataset[0].num_nodes
    assert 0 < num_nodes <= total_num_nodes
    node_index = torch.randperm(num_nodes)
    edge_index = internal_edges(dataset, node_index)
    num_train_nodes = int(train_frac * num_nodes)
    num_val_nodes = int(val_frac * num_nodes)
    train_index = node_index[:num_train_nodes]
    val_index = node_index[num_train_nodes: num_train_nodes + num_val_nodes]
    test_index = node_index[num_train_nodes + num_val_nodes:]
    train_mask = index_to_mask(total_num_nodes, train_index)
    val_mask = index_to_mask(total_num_nodes, val_index)
    test_mask = index_to_mask(total_num_nodes, test_index)
    return Data(
        x=dataset.x,
        edge_index=edge_index,
        y=dataset.y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=dataset.num_classes,
        num_features=dataset.num_features,
        name=dataset.name,
    )

class Cora:

    def __init__(self, root, split="sampled"):
        if split == "sampled":
            dataset = datasets.Planetoid(root=root, name='Cora')
            subset_size = dataset[0].num_nodes // 2
            target_set = sample_subgraph(dataset, subset_size)
            shadow_set = sample_subgraph(dataset, subset_size)
        elif split == "disjoint":
            dataset = datasets.Planetoid(root=root, name='Cora')
            num_nodes = dataset[0].num_nodes
            num_target_nodes = num_nodes // 2
            node_index = torch.randperm(num_nodes)
            target_index = node_index[:num_target_nodes]
            shadow_index = node_index[num_target_nodes:]
            target_set = extract_subgraph(dataset, target_index)
            shadow_set = extract_subgraph(dataset, shadow_index)
        elif split == "TSTF":
            # Deviating from the paper by Olatunji et al. by allowing overlap of target and shadow set.
            target_set = datasets.Planetoid(root=root, name='Cora', split='random', num_train_per_class=90)
            shadow_set = datasets.Planetoid(root=root, name='Cora', split='random', num_train_per_class=90)
        else:
            raise AttributeError("")
        self.target_set = target_set
        self.shadow_set = shadow_set

    def get_split(self):
        return self.target_set, self.shadow_set
