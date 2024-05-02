import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class GCN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = gnn.GCNConv(in_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, out_dim)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class GAT(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, heads=(8, 1), dropout=0.1):
        super(GAT, self).__init__()
        self.conv1 = gnn.GATConv(in_dim, hidden_dim, heads=heads[0], dropout=dropout)
        self.conv2 = gnn.GATConv(hidden_dim * heads[0], out_dim, heads=heads[1], dropout=dropout, concat=False)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    ''' 3 layer MLP net used for the attack model (binary classification). '''
    
    def __init__(self, in_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
