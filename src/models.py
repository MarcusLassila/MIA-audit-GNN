import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter

class BaseGNN(nn.Module):

    def __init__(self, dropout=0.0):
        super(BaseGNN, self).__init__()
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(input=x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GCN(BaseGNN):

    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0):
        super(GCN, self).__init__(dropout=dropout)
        channel_list = [in_dim, *hidden_dims, out_dim]
        self.convs = nn.ModuleList([
            gnn.GCNConv(in_c, out_c) for in_c, out_c in zip(channel_list, channel_list[1:])
        ])
        self.num_layers = len(self.convs)

class SGC(BaseGNN):

    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0):
        super(SGC, self).__init__(dropout=dropout)
        channel_list = [in_dim, *hidden_dims, out_dim]
        self.convs = nn.ModuleList([
            gnn.SGConv(in_c, out_c, K=2, cached=False) for in_c, out_c in zip(channel_list, channel_list[1:])
        ])
        self.num_layers = len(self.convs)

class GraphSAGE(BaseGNN):

    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0):
        super(GraphSAGE, self).__init__(dropout=dropout)
        channel_list = [in_dim, *hidden_dims, out_dim]
        self.convs = nn.ModuleList([
            gnn.SAGEConv(in_c, out_c, aggr="max") for in_c, out_c in zip(channel_list, channel_list[1:])
        ])
        self.num_layers = len(self.convs)

class GAT(BaseGNN):

    def __init__(self, in_dim, hidden_dims, out_dim, heads=(2,1), dropout=0.0):
        super(GAT, self).__init__(dropout=dropout)
        channel_list = [in_dim, *hidden_dims, out_dim]
        self.convs = nn.ModuleList([])
        for i, in_c, out_c, heads_prev, heads_curr in zip(range(len(channel_list)), channel_list, channel_list[1:], [1, *heads], heads):
            self.convs.append(gnn.GATConv(in_c * heads_prev, out_c, heads=heads_curr, concat=i+1<len(channel_list)))
        self.num_layers = len(self.convs)

class GIN(BaseGNN):

    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0):
        super(GIN, self).__init__(dropout=dropout)
        channel_list = [in_dim, *hidden_dims, out_dim]
        self.convs = nn.ModuleList([
            gnn.MLP(channel_list=[in_c, out_c, out_c]) for in_c, out_c in zip(channel_list, channel_list[1:])
        ])
        self.num_layers = len(self.convs)

class DecoupledGCN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0, num_layers=2):
        super(DecoupledGCN, self).__init__()
        self.num_layers = num_layers
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
        for _ in range(self.num_layers - 1):
            x = x[edge_index[1]]
            x = scatter(x, edge_index[0], dim=0, reduce='mean')
        x = scatter(x[edge_index[1]], edge_index[0], dim=0, reduce='mean')
        return x

class MLP(gnn.MLP):

    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0):
        channel_list = [in_dim, *hidden_dims, out_dim]
        super(MLP, self).__init__(channel_list, dropout=dropout)
        self.num_layers = len(channel_list) - 1
