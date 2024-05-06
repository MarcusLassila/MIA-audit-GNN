import torch
import torch_geometric
import torch_geometric.datasets as datasets
from torch_geometric.data import Data

torch.manual_seed(1)

def index_to_mask(size, index):
    mask = torch.zeros(size, dtype=bool)
    mask[index] = True
    return mask

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
    x = dataset.x
    y = dataset.y
    total_num_nodes = sum(graph.num_nodes for graph in dataset)
    index_set = set(node_index.numpy())
    edge_mask = torch.tensor([x.item() in index_set and y.item() in index_set for x, y in dataset.edge_index.t()])
    edge_index = dataset.edge_index[:,edge_mask]
    num_nodes = len(index_set)
    num_train_nodes = int(train_frac * num_nodes)
    num_val_nodes = int(val_frac * num_nodes)
    train_index = node_index[:num_train_nodes]
    val_index = node_index[num_train_nodes: num_train_nodes + num_val_nodes]
    test_index = node_index[num_train_nodes + num_val_nodes:]
    train_mask = index_to_mask(total_num_nodes, train_index)
    val_mask = index_to_mask(total_num_nodes, val_index)
    test_mask = index_to_mask(total_num_nodes, test_index)
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=dataset.num_classes,
        num_features=dataset.num_features,
        name=dataset.name,
    )

class Cora:

    def __init__(self, root, disjoint_split=False):
        if disjoint_split:
            dataset = datasets.Planetoid(root=root, name='Cora')
            num_nodes = sum(graph.num_nodes for graph in dataset)
            num_target_nodes = num_nodes // 2
            node_index = torch.randperm(num_nodes)
            target_index = node_index[:num_target_nodes]
            shadow_index = node_index[num_target_nodes:]
            self.target_dataset = extract_subgraph(dataset, target_index)
            self.shadow_dataset = extract_subgraph(dataset, shadow_index)
        else:
            self.target_dataset = datasets.Planetoid(root=root, name='Cora', split='random', num_train_per_class=90)
            self.shadow_dataset = datasets.Planetoid(root=root, name='Cora', split='random', num_train_per_class=90)

