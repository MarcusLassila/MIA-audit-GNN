import utils

import torch
import torch_geometric
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from sklearn.model_selection import train_test_split


def create_attack_dataset(shadow_dataset, shadow_model):
    features = shadow_model(shadow_dataset.x, shadow_dataset.edge_index).cpu()
    labels = shadow_dataset.train_mask.long().cpu()
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, stratify=labels)
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    return train_dataset, test_dataset

def train_split_interconnection_mask(dataset):
    mask = []
    for a, b in dataset.edge_index.T:
        mask.append(
            dataset.train_mask[a] == dataset.train_mask[b]
            and dataset.val_mask[a] == dataset.val_mask[b]
            and dataset.test_mask[a] == dataset.test_mask[b]
        )
    return torch.tensor(mask, dtype=torch.bool)

def masked_subgraph(graph, mask):
    '''
    Return the subgraph specified by the mask, keeping train, val, test and inductive masks of original graph.
    '''
    edge_index, _ = subgraph(
        subset=mask,
        edge_index=graph.edge_index,
        relabel_nodes=True,
    )
    data = Data(
        x=graph.x[mask],
        edge_index=edge_index,
        y=graph.y[mask],
        train_mask=graph.train_mask[mask],
        val_mask=graph.val_mask[mask],
        test_mask=graph.test_mask[mask],
        num_classes=graph.num_classes,
        num_features=graph.num_features,
    )
    data.inductive_mask = train_split_interconnection_mask(data)
    data.random_edge_mask = random_edge_mask(data)
    return data

def train_val_test_masks(num_nodes, train_frac, val_frac, stratify=None):
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if stratify is None:
        num_train_nodes = int(train_frac * num_nodes)
        num_val_nodes = int(val_frac * num_nodes)
        index = torch.randperm(num_nodes)
        train_index = index[:num_train_nodes]
        val_index = index[num_train_nodes: num_train_nodes + num_val_nodes]
        test_index = index[num_train_nodes + num_val_nodes:]
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
    else:
        bins = torch.unique(stratify)
        for b in bins:
            index = (stratify == b).nonzero().squeeze()
            perm_index = torch.randperm(index.shape[0])
            index = index[perm_index]
            num_train_nodes = int(train_frac * index.shape[0])
            num_val_nodes = int(val_frac * index.shape[0])
            train_mask[index[:num_train_nodes]] = True
            val_mask[index[num_train_nodes: num_train_nodes + num_val_nodes]] = True
            test_mask[index[num_train_nodes + num_val_nodes:]] = True
    return train_mask, val_mask, test_mask

def random_edge_mask(dataset):
    train_mask, _, test_mask = train_val_test_masks(dataset.x.shape[0], train_frac=0.5, val_frac=0.0, stratify=dataset.y)
    mask = []
    for a, b in dataset.edge_index.T:
        mask.append(
            train_mask[a] == train_mask[b]
            and test_mask[a] == test_mask[b]
        )
    return torch.tensor(mask, dtype=torch.bool)

def stochastic_block_model(root):
    block_sizes = [2000, 2000]
    edge_probs = torch.tensor([
        [1.0, 0.25],
        [0.25, 1.0],
    ]) * 0.001
    dataset = utils.execute_silently(
        callable=torch_geometric.datasets.StochasticBlockModelDataset, 
        root=root,
        block_sizes=block_sizes,
        edge_probs=edge_probs,
        num_channels=3,
        force_reload=True,
        class_sep=0.5,
    )
    train_mask, val_mask, test_mask = train_val_test_masks(
        num_nodes=sum(block_sizes),
        train_frac=0.4,
        val_frac=0.2,
        stratify=dataset.y,
    )
    data = Data(
        x=dataset.x,
        edge_index=dataset.edge_index,
        y=dataset.y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=2,
        num_features=3,
    )
    data.inductive_mask = train_split_interconnection_mask(data)
    data.name = "SBM"
    data.root = root
    return data

def extract_subgraph(dataset, node_index, train_frac=0.4, val_frac=0.2):
    '''
    Constructs a subgraph of dataset consisting of the nodes indexed in node_index with the edges linking them.
    Masks for training/validation/testing are constructed uniformly random with the specified proportions.
    '''
    if dataset.name == "SBM":
        # Temporary solution
        return stochastic_block_model(dataset.root)
    edge_index, _ = subgraph(
        subset=node_index,
        edge_index=dataset.edge_index,
        relabel_nodes=True,
        num_nodes=dataset.x.shape[0],
    )
    train_mask, val_mask, test_mask = train_val_test_masks(
        num_nodes=node_index.shape[0],
        train_frac=train_frac,
        val_frac=val_frac,
        stratify=dataset.y[node_index],
    )
    data = Data(
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
    data.inductive_mask = train_split_interconnection_mask(data)
    return data

def sample_subgraph(dataset, num_nodes, train_frac=0.4, val_frac=0.2, keep_class_proportions=True):
    '''
    Sample a subgraph by uniformly sample a number of nodes from the graph dataset.
    Masks for training/validation/testing are created uniformly at random with the specified proportions.
    The keep_class_proportions flag specifies that the sampled subgraph should have about the same
    proportions of nodes for each class as the full graph.
    '''
    total_num_nodes = dataset.x.shape[0]
    assert 0 < num_nodes <= total_num_nodes
    node_frac = num_nodes / total_num_nodes
    if keep_class_proportions:
        node_index = []
        for c in range(dataset.num_classes):
            index = (dataset.y == c).nonzero().squeeze()
            perm_index = torch.randperm(index.shape[0])
            index = index[perm_index]
            n = int(index.shape[0] * node_frac)
            node_index.append(index[:n])
        node_index = torch.cat(node_index)
    else:
        randomized_index = torch.randperm(total_num_nodes)
        node_index = randomized_index[:num_nodes]
    return extract_subgraph(dataset, node_index, train_frac=train_frac, val_frac=val_frac)

def disjoint_split(dataset, balance=0.5):
    '''
    Split the graph dataset in two disjoint subgraphs.
    The balance the fraction of nodes to use for the first subgraph, and the second subgraph gets the remaining nodes.
    '''
    node_index_A = []
    node_index_B = []
    for c in range(dataset.num_classes):
        index = (dataset.y == c).nonzero().squeeze()
        perm_index = torch.randperm(index.shape[0])
        index = index[perm_index]
        n = int(index.shape[0] * balance)
        node_index_A.append(index[:n])
        node_index_B.append(index[n:])
    node_index_A = torch.cat(node_index_A)
    node_index_B = torch.cat(node_index_B)
    return node_index_A, node_index_B

def target_shadow_split(dataset, split="sampled", target_frac=0.5, shadow_frac=0.5):
    if dataset.name == "SBM":
        # Temporary solution
        target_set = stochastic_block_model(dataset.root)
        shadow_set = stochastic_block_model(dataset.root)
        return target_set, shadow_set
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

def parse_dataset(root, name):
    match name:
        case "cora":
            dataset = torch_geometric.datasets.Planetoid(root=root, name="Cora")
        case "corafull":
            dataset = torch_geometric.datasets.CoraFull(root=root)
            dataset.name == "CoraFull"
        case"citeseer":
            dataset = torch_geometric.datasets.Planetoid(root=root, name="CiteSeer")
        case "chameleon":
            dataset = torch_geometric.datasets.WikipediaNetwork(root=root, name="chameleon")
        case "pubmed":
            dataset = torch_geometric.datasets.Planetoid(root=root, name="PubMed")
        case "flickr":
            dataset = torch_geometric.datasets.Flickr(root=root)
            dataset.name = "Flickr"
        case "sbm":
            dataset = stochastic_block_model(root)
        case _:
            raise ValueError("Unsupported dataset!")
    return dataset
