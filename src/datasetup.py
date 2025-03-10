import utils

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import degree, index_to_mask, mask_to_index, subgraph
from collections import deque

def merge_graphs(graph_A, graph_B):
    x = torch.concat([graph_A.x, graph_B.x])
    y = torch.concat([graph_A.y, graph_B.y])
    edge_index = torch.concat([graph_A.edge_index, graph_B.edge_index + graph_A.num_nodes], dim=1)
    train_mask = torch.zeros(graph_A.num_nodes + graph_B.num_nodes, dtype=torch.bool)
    train_mask[mask_to_index(graph_A.train_mask)] = True
    train_mask[graph_A.num_nodes + mask_to_index(graph_B.train_mask)] = True
    val_mask = torch.zeros(graph_A.num_nodes + graph_B.num_nodes, dtype=torch.bool)
    val_mask[mask_to_index(graph_A.val_mask)] = True
    val_mask[graph_A.num_nodes + mask_to_index(graph_B.val_mask)] = True
    test_mask = torch.zeros(graph_A.num_nodes + graph_B.num_nodes, dtype=torch.bool)
    test_mask[mask_to_index(graph_A.test_mask)] = True
    test_mask[graph_A.num_nodes + mask_to_index(graph_B.test_mask)] = True
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=graph_A.num_classes,
    )
    data.inductive_mask = train_split_interconnection_mask(data)
    return data

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
    )
    data.inductive_mask = train_split_interconnection_mask(data)
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
            index = (stratify == b).nonzero().flatten()
            perm_index = torch.randperm(index.shape[0])
            index = index[perm_index]
            num_train_nodes = int(train_frac * index.shape[0])
            num_val_nodes = int(val_frac * index.shape[0])
            train_mask[index[:num_train_nodes]] = True
            val_mask[index[num_train_nodes: num_train_nodes + num_val_nodes]] = True
            test_mask[index[num_train_nodes + num_val_nodes:]] = True
    return train_mask, val_mask, test_mask

def random_edge_mask(dataset, frac=0.5):
    ''' Return a mask that uniformly randomly masks out a specified fraction of the edges. '''
    mask = torch.ones(dataset.edge_index.shape[1], dtype=torch.bool)
    index = np.random.choice(mask.shape[0], int(frac * mask.shape[0]), replace=False)
    mask[index] = False
    return mask

def stochastic_block_model(root):
    block_sizes = [2000, 2000]
    edge_probs = torch.tensor([
        [1.0, 0.25],
        [0.25, 1.0],
    ]) * 0.005
    num_features = 4
    dataset = utils.execute_silently(
        callable=torch_geometric.datasets.StochasticBlockModelDataset, 
        root=root,
        block_sizes=block_sizes,
        edge_probs=edge_probs,
        num_channels=num_features,
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
    )
    data.inductive_mask = train_split_interconnection_mask(data)
    data.name = "SBM"
    data.root = root
    return data

def node_index_complement(node_index, num_nodes):
    return mask_to_index(~index_to_mask(node_index, size=num_nodes))

def sample_nodes(total_num_nodes, num_sampled_nodes, stratify=None):
    if stratify is None:
        node_index = torch.randperm(total_num_nodes)[:num_sampled_nodes]
    else:
        node_index = []
        node_frac = num_sampled_nodes / total_num_nodes
        for c in torch.unique(stratify):
            index = (stratify == c).nonzero().flatten()
            perm_index = torch.randperm(index.shape[0])
            index = index[perm_index]
            n = int(index.shape[0] * node_frac)
            node_index.append(index[:n])
        node_index = torch.cat(node_index)
    return node_index

def sample_nodes_v2(dataset, num_nodes, num_neighbors):
    node_index = sample_nodes(dataset.num_nodes, num_nodes, stratify=dataset.y)
    node_index = node_index[torch.randperm(node_index.shape[0])]
    edge_index = dataset.edge_index

    def sample_local_neighborhood(central_node, visited):
        queue = deque([(0, central_node.item())])
        while queue:
            i, u = queue.popleft()
            if i >= len(num_neighbors):
                break
            if u in visited:
                continue
            visited.add(u)
            edge_mask = edge_index[0] == u
            neighbors = edge_index[1][edge_mask]
            perm_mask = torch.randperm(neighbors.shape[0])
            neighbors = neighbors[perm_mask]
            for v in neighbors.tolist():
                queue.append((i + 1, v))

    sampled_nodes = set()
    for node in node_index:
        if node in sampled_nodes:
            continue
        sample_local_neighborhood(node, sampled_nodes)
        if len(sampled_nodes) >= num_nodes:
            break

    sampled_node_index = torch.tensor(list(sampled_nodes), dtype=torch.long)[:num_nodes]
    return sampled_node_index

def random_walk(edge_index, available, priority, path_length):
    visited = set()
    start_node = priority.pop()
    while start_node not in available:
       start_node = priority.pop()
    queue = deque([start_node])
    while queue:
        u = queue.popleft()
        if u in visited:
            continue
        visited.add(u)
        available.remove(u)
        if len(visited) == path_length:
            break
        edge_mask = edge_index[0] == u
        neighbors = edge_index[1][edge_mask]
        perm_mask = torch.randperm(neighbors.shape[0])
        neighbors = neighbors[perm_mask]
        for v in neighbors.tolist():
            if v in available:
                queue.append(v)
    return visited, available

def random_walk_node_sample(dataset, num_nodes):
    edge_index = dataset.edge_index
    #_, sorted_node_index = degree(edge_index[0], num_nodes=dataset.x.shape[0], dtype=torch.long).sort(descending=False)
    priority = [x.item() for x in torch.randperm(dataset.x.shape[0])] # [x.item() for x in sorted_node_index]
    node_index_set = set()
    available = set(range(dataset.x.shape[0]))
    while len(node_index_set) < num_nodes:
        node_index, available = random_walk(edge_index, available, priority, path_length=50)
        node_index_set.update(node_index)
    return torch.tensor([*node_index_set])[:num_nodes]

def random_walk_alt(edge_index, available, path_length):
    available_set = set(available)
    visited = set()
    start_node = available[-1]
    queue = deque([start_node])
    while queue:
        u = queue.popleft()
        if u in visited:
            continue
        visited.add(u)
        available_set.remove(u)
        if len(visited) == path_length:
            break
        edge_mask = edge_index[0] == u
        neighbors = edge_index[1][edge_mask]
        perm_mask = torch.randperm(neighbors.shape[0])
        neighbors = neighbors[perm_mask]
        for v in neighbors.tolist():
            if v in available_set:
                queue.append(v)
    available = [x for x in available if x in available_set]
    return visited, available

def alternating_random_walk_node_split(dataset):
    edge_index = dataset.edge_index
    _, indices = degree(edge_index[0], num_nodes=dataset.num_nodes, dtype=torch.long).sort(descending=True)
    available = [x.item() for x in indices]

    node_index_split = set(), set()
    turn = 0
    while available:
        node_index, available = random_walk_alt(edge_index, available, path_length=50)
        node_index_split[turn].update(node_index)
        turn ^= 1

    node_index_A = torch.tensor(list(node_index_split[0]), dtype=torch.long)
    node_index_B = torch.tensor(list(node_index_split[1]), dtype=torch.long)
    return node_index_A, node_index_B

def remasked_graph(graph, train_frac, val_frac, stratify=None):
    '''
    Return a copy of dataset with new train/val/test masks.
    '''
    train_mask, val_mask, test_mask = train_val_test_masks(
        num_nodes=graph.num_nodes,
        train_frac=train_frac,
        val_frac=val_frac,
        stratify=stratify,
    )
    data = graph.clone()
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.inductive_mask = train_split_interconnection_mask(data)
    return data

def remasked_graph_deterministic(graph, train_mask):
    data = graph.clone()
    data.train_mask = train_mask
    data.test_mask = ~train_mask
    data.val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    data.inductive_mask = train_split_interconnection_mask(data)
    return data

def add_masks(graph, train_frac, val_frac, stratify=None):
    '''
    Mutate graph to add new train/val/test masks.
    '''
    train_mask, val_mask, test_mask = train_val_test_masks(
        num_nodes=graph.num_nodes,
        train_frac=train_frac,
        val_frac=val_frac,
        stratify=stratify,
    )
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    graph.inductive_mask = train_split_interconnection_mask(graph)

def extract_subgraph(graph, node_index, train_frac, val_frac):
    '''
    Constructs a subgraph of dataset consisting of the nodes indexed in node_index with the edges linking them.
    Masks for training/validation/testing are constructed uniformly random with the specified proportions.
    '''
    edge_index, _ = subgraph(
        subset=node_index,
        edge_index=graph.edge_index,
        relabel_nodes=True,
        num_nodes=graph.num_nodes,
    )
    train_mask, val_mask, test_mask = train_val_test_masks(
        num_nodes=node_index.shape[0],
        train_frac=train_frac,
        val_frac=val_frac,
        stratify=graph.y[node_index],
    )
    data = Data(
        x=graph.x[node_index],
        edge_index=edge_index,
        y=graph.y[node_index],
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=graph.num_classes,
    )
    data.inductive_mask = train_split_interconnection_mask(data)
    return data

def sample_subgraph(dataset, num_nodes, train_frac, val_frac, v2=False):
    '''
    Sample a subgraph by uniformly sample a number of nodes from the graph dataset.
    Masks for training/validation/testing are created uniformly at random with the specified proportions.
    The keep_class_proportions flag specifies that the sampled subgraph should have about the same
    proportions of nodes for each class as the full graph.
    '''
    assert 0 < num_nodes <= dataset.num_nodes
    if v2:
        node_index = sample_nodes_v2(dataset, num_nodes, num_neighbors=[5, 5])
    else:
        node_index = sample_nodes(dataset.num_nodes, num_nodes, stratify=dataset.y)
    return extract_subgraph(dataset, node_index, train_frac=train_frac, val_frac=val_frac)

def disjoint_node_split(dataset, v2=False):
    '''
    Split the nodes into two disjoint sets.
    Return node index tensors for the two sets.
    '''
    num_nodes_A = dataset.num_nodes // 2
    if v2:
        node_index_A, node_index_B = alternating_random_walk_node_split(dataset) # Only supports balance = 0.5
    else:
        node_index_A = sample_nodes(dataset.num_nodes, num_nodes_A, stratify=dataset.y)
        node_index_B = node_index_complement(node_index_A, dataset.num_nodes)
    return node_index_A, node_index_B

def disjoint_graph_split(graph, train_frac, val_frac, v2=True):
    '''
    Split the graph dataset in two rougly equal sized disjoint subgraphs.
    '''
    index_A, index_B = disjoint_node_split(graph, v2=v2)
    graph_A = extract_subgraph(graph, index_A, train_frac=train_frac, val_frac=val_frac)
    graph_B = extract_subgraph(graph, index_B, train_frac=train_frac, val_frac=val_frac)
    return graph_A, graph_B, index_A, index_B

def parse_dataset(root, name, max_num_nodes=None):
    match name:
        case "amazon-computers":
            dataset = torch_geometric.datasets.Amazon(root=f'{root}/Amazon', name="Computers")
        case "amazon-photo":
            dataset = torch_geometric.datasets.Amazon(root=f'{root}/Amazon', name="Photo")
        case "chameleon":
            dataset = torch_geometric.datasets.WikipediaNetwork(root=root, name="chameleon")
        case"citeseer":
            dataset = torch_geometric.datasets.Planetoid(root=root, name="CiteSeer")
        case "cora":
            dataset = torch_geometric.datasets.Planetoid(root=root, name="Cora")
        case "corafull":
            dataset = torch_geometric.datasets.CoraFull(root=f'{root}/CoraFull')
        case "flickr":
            dataset = torch_geometric.datasets.Flickr(root=f'{root}/Flickr')
        case "github":
            dataset = torch_geometric.datasets.GitHub(root=f'{root}/GitHub')
        case "pubmed":
            dataset = torch_geometric.datasets.Planetoid(root=root, name="PubMed")
        case "reddit":
            dataset = torch_geometric.datasets.Reddit(root=f'{root}/Reddit')
        case "sbm":
            dataset = stochastic_block_model(root)
        case _:
            raise ValueError("Unsupported dataset!")
    if max_num_nodes is not None and dataset.x.shape[0] > max_num_nodes:
        print(f'Sampling a subgraph of {name}.', flush=True)
        nodes = random_walk_node_sample(dataset, max_num_nodes)
        edge_index, _ = subgraph(
            subset=nodes,
            edge_index=dataset.edge_index,
            relabel_nodes=True,
        )
        dataset = Data(
            x=dataset.x[nodes],
            edge_index=edge_index,
            y=dataset.y[nodes],
            num_classes=dataset.num_classes,
        )
    else:
        dataset = Data(
            x=dataset.x,
            edge_index=dataset.edge_index,
            y=dataset.y,
            num_classes=dataset.num_classes,
        )
    return dataset

def test_split():
    from statistics import mean, stdev
    from tqdm.auto import tqdm
    dataset = parse_dataset('./data', 'amazon-photo')
    print(dataset)
    degree_A = []
    degree_B = []
    fractions_A = []
    fractions_B = []
    for _ in tqdm(range(20), desc='Computing datasplit statistics'):
        data_A, data_B = disjoint_graph_split(dataset, v2=False)
        average_degree_A, _, _ = utils.average_degree(data_A)
        average_degree_B, _, _ = utils.average_degree(data_B)
        frac_A = utils.fraction_isolated_nodes(data_A)
        frac_B = utils.fraction_isolated_nodes(data_B)
        degree_A.append(average_degree_A)
        degree_B.append(average_degree_B)
        fractions_A.append(frac_A)
        fractions_B.append(frac_B)
    s = (
        f'average degree A (mean, std): ({mean(degree_A):.4f}, {stdev(degree_A):.4f})\n'
        f'average degree B (mean, std): ({mean(degree_B):.4f}, {stdev(degree_B):.4f})\n'
        f'fraction isolated A (mean, std): ({mean(fractions_A):.4f}, {stdev(fractions_A):.4f})\n'
        f'fraction isolated B (mean, std): ({mean(fractions_B):.4f}, {stdev(fractions_B):.4f})\n'
    )
    print(s)

if __name__ == '__main__':
    test_split()
    pass
