import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

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
        return F.log_softmax(x, dim=1)

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
        self.conv1 = gnn.GINConv(MLP(in_dim, (hidden_dim,), hidden_dim))
        self.conv2 = gnn.GINConv(MLP(hidden_dim, (hidden_dim,), out_dim))

class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dims, out_dim=2):
        super(MLP, self).__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        self.layers = nn.ModuleList([nn.Linear(x, y) for x, y in zip(dims, dims[1:])])
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x
