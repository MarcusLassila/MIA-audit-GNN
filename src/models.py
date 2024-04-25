import torch
import torch.nn.functional as F
import torch_geometric.nn as nn

class GCN(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = nn.GCNConv(in_dim, hidden_dim)
        self.conv2 = nn.GCNConv(hidden_dim, out_dim)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
