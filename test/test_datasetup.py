import sys
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_dir, '..', 'src'))

from src import datasetup
import unittest

class TestInductiveSplit(unittest.TestCase):
    
    def test_inductive_mask(self):
        graph = datasetup.parse_dataset(os.path.join(curr_dir, '..', 'data'), 'cora')
        graph = datasetup.new_train_split_mask(graph)
        for a, b in graph.edge_index[:, graph.inductive_mask].T:
            self.assertEqual(graph.train_mask[a], graph.train_mask[b])
            self.assertEqual(graph.val_mask[a], graph.val_mask[b])
            self.assertEqual(graph.test_mask[a], graph.test_mask[b])

if __name__ == '__main__':
    unittest.main()
