from math import ceil
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GINConv, MLP, DenseGINConv, PANConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import TopKPooling, PANPooling, SAGPooling, ASAPooling, EdgePooling, graclus
from torch_geometric.nn import dense_mincut_pool, dense_diff_pool, DMoNPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_adj, to_dense_batch

class Base_GIN_Net(torch.nn.Module):
    def __init__(self, 
                 in_channels,           # Size of node features
                 out_channels,          # Number of classes
                 num_layers_pre=1,      # Number of GIN layers before pooling
                 num_layers_post=1,     # Number of GIN layers after pooling
                 hidden_channels=64,    # Dimensionality of node embeddings
                 norm=True,             # Normalise Layers in the GIN MLP
                 activation='ELU',      # Activation of the MLP in GIN 
                 ):
        super(Base_GIN_Net, self).__init__()

        self.num_layers_pre = num_layers_pre
        self.num_layers_post = num_layers_post
        self.hidden_channels = hidden_channels
        self.act = activation_resolver(activation)

        self.conv_layers_pre = torch.nn.ModuleList()
        for _ in range(num_layers_pre):
            mlp = MLP([in_channels, hidden_channels, hidden_channels], act=activation)
            self.conv_layers_pre.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.conv_layers_post = torch.nn.ModuleList()
        for _ in range(num_layers_post):
            mlp = MLP([hidden_channels, hidden_channels, hidden_channels], act=activation, norm=None)
            self.conv_layers_post.append(GINConv(nn=mlp, train_eps=False))

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

    def forward(self, data):
        x = data.x    
        adj = data.edge_index
        batch = data.batch

        for layer in self.conv_layers_pre:  
            x = self.act(layer(x, adj))

        for layer in self.conv_layers_post:  
            x = self.act(layer(x, adj))

        x = global_add_pool(x, batch)
        x = self.mlp(x)
        aux_loss=0
        return F.log_softmax(x, dim=-1), aux_loss