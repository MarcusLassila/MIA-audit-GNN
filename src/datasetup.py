import torch
import torch_geometric
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask, subgraph
from sklearn.model_selection import train_test_split


def create_attack_dataset(shadow_dataset, shadow_model):
    features = shadow_model(shadow_dataset.x, shadow_dataset.edge_index).cpu()
    labels = shadow_dataset.train_mask.long().cpu()
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, stratify=labels)
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    return train_dataset, test_dataset

def train_split_interconnection_mask(graph):
    mask = []
    for a, b in graph.edge_index.T:
        mask.append(
            graph.train_mask[a] == graph.train_mask[b]
            and graph.val_mask[a] == graph.val_mask[b]
            and graph.test_mask[a] == graph.test_mask[b]
        )
    return torch.tensor(mask, dtype=torch.bool)

def extract_subgraph(dataset, node_index, train_frac=0.4, val_frac=0.2):
    '''
    Constructs a subgraph of dataset consisting of the nodes indexed in node_index with the edges linking them.
    Masks for training/validation/testing are constructed uniformly random with the specified proportions.
    '''
    edge_index, _ = subgraph(
        subset=node_index,
        edge_index=dataset.edge_index,
        relabel_nodes=True,
        num_nodes=dataset.x.shape[0],
    )
    num_nodes = len(node_index)
    num_train_nodes = int(train_frac * num_nodes)
    num_val_nodes = int(val_frac * num_nodes)
    randomized_index = torch.randperm(num_nodes)
    train_index = randomized_index[:num_train_nodes]
    val_index = randomized_index[num_train_nodes: num_train_nodes + num_val_nodes]
    test_index = randomized_index[num_train_nodes + num_val_nodes:]
    train_mask = index_to_mask(train_index, num_nodes)
    val_mask = index_to_mask(val_index, num_nodes)
    test_mask = index_to_mask(test_index, num_nodes)
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
    inductive_mask = train_split_interconnection_mask(data)
    data.inductive_mask = inductive_mask
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
            perm_mask = torch.randperm(index.shape[0])
            index = index[perm_mask]
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

def parse_dataset(root, name, sbm_parameters=None):
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
            if sbm_parameters:
                block_sizes, edge_probs, num_channels = sbm_parameters
            else:
                block_sizes = torch.tensor([500, 500], dtype=torch.long)
                edge_probs = torch.tensor([
                    [0.3, 0.1],
                    [0.1, 0.3],
                ])
                num_channels = 4
            dataset = torch_geometric.datasets.StochasticBlockModelDataset(
                root=root,
                block_sizes=block_sizes,
                edge_probs=edge_probs,
                num_channels=num_channels,
                force_reload=False,
            )
            dataset.name = "SBM"
        case _:
            raise ValueError("Unsupported dataset!")
    return dataset
