import sys
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_dir, '..', 'src'))

from src import utils
import unittest
import torch

class TestDataSetup(unittest.TestCase):
    
    def test_partition_training_set(self):
        num_nodes = 7541
        num_models = 128
        masks = utils.partition_training_sets(num_nodes=num_nodes, num_models=num_models)
        self.assertEqual(masks.shape[0], num_models, 'First dim of training partition masks should be equal to the number of models')
        self.assertEqual(masks.shape[1], num_nodes, 'Second dim of training partition masks should be equal to the number of nodes')
        self.assertEqual(masks.dtype, torch.bool, 'Training partition masks should be boolean valued')
        for i in range(0, 128, 2):
            self.assertFalse(torch.any(masks[i] & masks[i + 1]), 'Mask 2i and 2i+1 should not overlap')
            self.assertEqual(torch.sum(masks[i] | masks[i + 1]), num_nodes, 'The union of mask 2i and 2i+1 should include all nodes')
            self.assertLessEqual(torch.abs(masks[i].sum() - masks[i + 1].sum()), 1, 'Mask 2i and 2i+1 should include half of the nodes')

if __name__ == '__main__':
    unittest.main()
