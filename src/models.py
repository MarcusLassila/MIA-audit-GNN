import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter

class TwoLayerGNN(nn.Module):
    ''' Base class for the different GNN architectures to inherit common implementations from. '''

    def __init__(self, dropout=0.0):
        super(TwoLayerGNN, self).__init__()
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(input=x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCN(TwoLayerGNN):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super(GCN, self).__init__(dropout=dropout)
        self.conv1 = gnn.GCNConv(in_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, out_dim)

class SGC(TwoLayerGNN):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super(SGC, self).__init__(dropout=dropout)
        self.conv1 = gnn.SGConv(in_dim, hidden_dim, K=2, cached=False)
        self.conv2 = gnn.SGConv(hidden_dim, out_dim, K=2, cached=False)

class GraphSAGE(TwoLayerGNN):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super(GraphSAGE, self).__init__(dropout=dropout)
        self.conv1 = gnn.SAGEConv(in_dim, hidden_dim)
        self.conv2 = gnn.SAGEConv(hidden_dim, out_dim)

class GAT(TwoLayerGNN):

    def __init__(self, in_dim, hidden_dim, out_dim, heads=(8, 1), dropout=0.0):
        super(GAT, self).__init__(dropout=dropout)
        self.conv1 = gnn.GATConv(in_dim, hidden_dim, heads=heads[0], dropout=dropout)
        self.conv2 = gnn.GATConv(hidden_dim * heads[0], out_dim, heads=heads[1], dropout=dropout, concat=False)

class GIN(TwoLayerGNN):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super(GIN, self).__init__(dropout=dropout)
        self.conv1 = gnn.GINConv(gnn.MLP(channel_list=[in_dim, hidden_dim, hidden_dim]))
        self.conv2 = gnn.GINConv(gnn.MLP(channel_list=[hidden_dim, hidden_dim, out_dim]))

class DecoupledGCN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0, num_propagations=2):
        super(DecoupledGCN, self).__init__()
        self.num_propagations = num_propagations
        self.mlp = gnn.MLP(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=1,
            dropout=dropout,
        )

    def forward(self, x, edge_index):
        x = self.mlp(x)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        for _ in range(self.num_propagations - 1):
            x = x[edge_index[1]]
            x = scatter(x, edge_index[0], dim=0, reduce='mean')
        x = scatter(x[edge_index[1]], edge_index[0], dim=0, reduce='mean')
        return x
