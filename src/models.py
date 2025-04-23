import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class BaseGNN(nn.Module):

    def __init__(self, dropout=0.0):
        super(BaseGNN, self).__init__()
        self.dropout = dropout
        self._dropout_during_inference = False

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    @property
    def dropout_during_inference(self):
        return self._dropout_during_inference

    @dropout_during_inference.setter
    def dropout_during_inference(self, value):
        self._dropout_during_inference = value

    def forward(self, x, edge_index=None):
        if edge_index is None:
            x, edge_index = x
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(input=x, p=self.dropout, training=self.training or self._dropout_during_inference)
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

    def __init__(self, in_dim, hidden_dims, out_dim, heads=(8,1), dropout=0.0):
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

class MLP(gnn.MLP):

    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0):
        channel_list = [in_dim, *hidden_dims, out_dim]
        super(MLP, self).__init__(channel_list, dropout=dropout)

class XMLP(nn.Module):
    ''' Attack model from Xinlei He et al. that pass the concatenated query to separate linear layers before applying the MLP '''

    def __init__(self, concats, in_dim, hidden_dims, out_dim, dropout=0.0):
        super(XMLP, self).__init__()
        self.concats = concats
        self.in_dim = in_dim
        self.concat_linear_embedding_dim = hidden_dims[0]
        self.lin_layers = nn.ModuleList([
            nn.Linear(in_features=in_dim, out_features=hidden_dims[0]) for _ in range(concats)
        ])
        channel_list = [concats * hidden_dims[0], *hidden_dims[1:], out_dim]
        self.mlp = gnn.MLP(channel_list, dropout=dropout)

    def forward(self, x):
        x = x.view(-1, self.concats, self.in_dim).permute(1, 0, 2)
        x = torch.cat([lin_layer(z) for lin_layer, z in zip(self.lin_layers, x)], dim=1)
        x = self.mlp(x)
        return x