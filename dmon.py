import torch
import torch.nn.functional as F
from math import ceil
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import GINConv, DMoNPooling,MLP,DenseGINConv, Linear

class DMonGIN(torch.nn.Module):
    def __init__(self, 
                 in_channels,           # Size of node features
                 out_channels,          # Number of classes
                 num_layers_pre=1,      # Number of GIN layers before pooling
                 num_layers_post=1,     # Number of GIN layers after pooling
                 hidden_channels=64,    # Dimensionality of node embeddings
                 norm=True,             # Normalise Layers in the GIN MLP
                 activation='ELU',      # Activation of the MLP in GIN 
                 average_nodes=None,    # Needed for dense pooling methods
                 pool_ratio=0.1,        # Ratio = nodes_after_pool/nodes_before_pool
                 ):
        super(DMonGIN, self).__init__()

        self.num_layers_pre = num_layers_pre
        self.num_layers_post = num_layers_post
        self.hidden_channels = hidden_channels
        self.act = activation_resolver(activation)
        self.pool_ratio = pool_ratio

        # Pre-pooling block
        self.conv_layers_pre = torch.nn.ModuleList()
        for _ in range(num_layers_pre):
            mlp = MLP([in_channels, hidden_channels, hidden_channels], act=activation)
            self.conv_layers_pre.append(GINConv(nn=mlp,train_eps=False))
            in_channels = hidden_channels

        # Pooling block
        pooled_nodes = ceil(pool_ratio*average_nodes)
        self.pool = DMoNPooling(hidden_channels, pooled_nodes)

        # Post-pooling block
        self.conv_layers_post = torch.nn.ModuleList()
        for _ in range(num_layers_post):
            mlp = MLP([hidden_channels, hidden_channels, hidden_channels], act=activation, norm=None)
            self.conv_layers_post.append(DenseGINConv(nn=mlp, train_eps=False))

        # Readout
        self.mlp = MLP([hidden_channels, hidden_channels, hidden_channels//2, out_channels],
                       act=activation,
                       norm=None,
                       dropout=0.5)
        
    def reset_parameters(self):
        for (name, module) in self._modules.items():
            try:
                module.reset_parameters()
            except AttributeError:
                if name != 'act':
                    for x in module:
                        x.reset_parameters()

    def forward(self,data):
        x = data.x
        adj = data.edge_index
        batch = data.batch

        ### pre-pooling block 
        for layer in self.conv_layers_pre:
            x = self.act(layer(x, adj))

        ### pooling block
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(adj, batch)
        _, x, adj, l1, l2, l3 = self.pool(x, adj, mask)
        aux_loss = 0.3*l1 + 0.3*l2 + 0.3*l3

        ### post-pooling block
        for layer in self.conv_layers_post:
            x = self.act(layer(x, adj))

        ### readout
        x = torch.sum(x, dim=1)
        x = self.mlp(x)

        return F.log_softmax(x, dim=-1), aux_loss
    

        