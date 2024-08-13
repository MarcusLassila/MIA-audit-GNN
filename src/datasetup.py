import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask, subgraph
from sklearn.model_selection import train_test_split

global_variables = {
    'transductive': False,
    'simplified_dataset': False,
}

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

def remove_train_val_test_interconnections(graph):
    mask = []
    for a, b in graph.edge_index.T:
        mask.append(
            graph.train_mask[a] == graph.train_mask[b]
            and graph.val_mask[a] == graph.val_mask[b]
            and graph.test_mask[a] == graph.test_mask[b]
        )
    graph.edge_index = graph.edge_index.T[mask].T

def extract_subgraph(dataset, node_index, train_frac=0.5, val_frac=0.2):
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
    if not global_variables['transductive']:
        remove_train_val_test_interconnections(data)
    if global_variables['simplified_dataset']:
        return simplified_dataset(data)
    else:
        return data

def sample_subgraph(dataset, num_nodes, train_frac=0.5, val_frac=0.2, keep_class_proportions=True):
    '''
    Sample a subgraph by uniformly sample a number of nodes from the graph dataset.
    Masks for training/validation/testing are created uniformly at random with the specified proportions.
    The keep_class_proportions flag specifies that the sampled subgraph should have about the same
    proportions of nodes for each class as the full graph.
    '''
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
        case _:
            raise ValueError("Unsupported dataset!")
    return dataset

def simplified_dataset(dataset, num_features=4, noise_std=0.0):
    '''
    Combine classes into two new classes and replaces feature vectors according to
    ______|  Class 0  |  Class 1  |
    Train | [1,0,0,0] | [0,1,0,0] |
    Test  | [0,0,1,0] | [0,0,0,1] |
    ______|___________|___________|
    Gaussian noise is added to the feature vectors with mean=0 and std=noise_std.
    '''
    a = (dataset.num_classes + 1) // 2
    b = num_features // 4
    y = (dataset.y >= a).long()
    x = torch.zeros(size=(dataset.x.shape[0], num_features))
    label_mask = y == 0
    train_mask = ~dataset.test_mask
    x[train_mask & label_mask, :b] = 1.0
    x[train_mask & ~label_mask, b: 2 * b] = 1.0
    x[~train_mask & label_mask, 2 * b: 3 * b] = 1.0
    x[~train_mask & ~label_mask, 3 * b:] = 1.0
    x = x + torch.normal(mean=0.0, std=noise_std, size=x.shape)
    return Data(
        x=x,
        edge_index=dataset.edge_index,
        y=y,
        train_mask=dataset.train_mask,
        val_mask=dataset.val_mask,
        test_mask=dataset.test_mask,
        num_classes=2,
        num_features=num_features,
        name=dataset.name,
    )
