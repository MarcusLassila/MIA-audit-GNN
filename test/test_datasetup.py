import sys
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_dir, '..', 'src'))

from src import datasetup
import unittest
import torch

class TestDataSetup(unittest.TestCase):
    
    def test_inductive_mask(self):
        graph = datasetup.parse_dataset(os.path.join(curr_dir, '..', 'data'), 'cora')
        graph = datasetup.random_remasked_graph(graph, train_frac=0.4, val_frac=0.2)
        for a, b in graph.edge_index[:, graph.inductive_mask].T:
            self.assertEqual(graph.train_mask[a], graph.train_mask[b])
            self.assertEqual(graph.val_mask[a], graph.val_mask[b])
            self.assertEqual(graph.test_mask[a], graph.test_mask[b])

    def test_disjoint_node_split(self):
        graph = datasetup.parse_dataset(os.path.join(curr_dir, '..', 'data'), 'cora')
        for v2 in False, True:
            node_index_A, node_index_B = datasetup.disjoint_node_split(graph, v2=v2)
            overlap = set(node_index_A).intersection(set(node_index_B))
            self.assertEqual(len(overlap), 0, 'Nodes are not disjoint')

    def test_extract_subgraph(self):
        graph = datasetup.parse_dataset(os.path.join(curr_dir, '..', 'data'), 'cora')
        num_nodes = 500
        node_index = torch.arange(graph.num_nodes)[torch.randperm(graph.num_nodes)][:num_nodes]
        subgragh = datasetup.extract_subgraph(graph, node_index, train_frac=0.5, val_frac=0.0)
        self.assertTrue(torch.equal(graph.x[node_index], subgragh.x), 'Node features are incorrect')
        self.assertTrue(torch.equal(graph.y[node_index], subgragh.y), 'Node labels are incorrect')
        self.assertEqual(subgragh.val_mask.sum().item(), 0, 'Validation mask is nonzero')

if __name__ == '__main__':
    unittest.main()
