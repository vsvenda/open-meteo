# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:16:40 2024

@author: Ana
"""

from torch_geometric.nn import TransformerConv
import torch
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_linear_layers,
                 dropout, beta=True, heads=1):
        """
        Params:
        - input_dim: The dimension of input features for each node.
        - hidden_dim: The size of the hidden layers.
        - output_dim: The dimension of the output features (often equal to the
            number of classes in a classification task).
        - num_layers: The number of layer blocks in the model.
        - dropout: The dropout rate for regularization. It is used to prevent
            overfitting, helping the learning process remains generalized.
        - beta: A boolean parameter indicating whether to use a gated residual
            connection (based on equations 5 and 6 from the UniMP paper). The
            gated residual connection (controlled by the beta parameter) helps
            preventing overfitting by allowing the model to balance between new
            and existing node features across layers.
        - heads: The number of heads in the multi-head attention mechanism.
        """
        super(GNN, self).__init__()
  
        # The list of transormer conv layers for the each layer block.
        self.num_layers = num_layers
        conv_layers = [TransformerConv(input_dim, hidden_dim//heads, heads=heads, beta=beta)]
        conv_layers += [TransformerConv(hidden_dim, hidden_dim//heads, heads=heads, beta=beta) for _ in range(num_layers - 2)]
        # In the last layer, we will employ averaging for multi-head output by
        # setting concat to True.
        #without linear layer
        # conv_layers.append(TransformerConv(hidden_dim, output_dim, heads = heads, beta=beta, concat=False))
        self.convs = torch.nn.ModuleList(conv_layers)
  
        # The list of layerNorm for each layer block.
        norm_layers = [torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)
        
        #with linear output
        layers = []
        
        # First layer to expand from hidden_dim to hidden_dim * 2
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim * 2))
        layers.append(torch.nn.ReLU())
        
        # Intermediate layers
        for _ in range(num_linear_layers - 2):
            layers.append(torch.nn.Linear(hidden_dim * 2, hidden_dim * 2))
            layers.append(torch.nn.ReLU())
        
        # Final layer to map to output_dim
        layers.append(torch.nn.Linear(hidden_dim * 2, output_dim))
        
        # Combine layers into a Sequential module
        #with linear
        self.linear_output = torch.nn.Sequential(*layers)
        
        self.out_channels = output_dim
        # Probability of an element getting zeroed.
        self.dropout = dropout
        
    def reset_parameters(self):
        """
        Resets the parameters of the convolutional and normalization layers,
        ensuring they are re-initialized when needed.
        """
        for conv in self.convs:
            conv.reset_parameters()
        
        for norm in self.norms:
            norm.reset_parameters()
            
        for layer in self.linear_output:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(self, x, edge_index):
        """
        The input features are passed sequentially through the transformer
        convolutional layers. After each convolutional layer (except the last),
        the following operations are applied:
        - Layer normalization (`LayerNorm`).
        - ReLU activation function.
        - Dropout for regularization.
        The final layer is processed without layer normalization and ReLU
        to average the multi-head results for the expected output.
    
        Params:
        - x: node features x
        - edge_index: edge indices.
    
        """
        x = self.convs[0](x, edge_index)
        x = self.norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout, training = self.training)
        for i in range(1, self.num_layers - 1):
          # Construct the network as shown in the model architecture.
            residual = x
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p = self.dropout, training = self.training)
            x = x + residual
    
        # Last layer, average multi-head output.
        # with last conv layer
        # x = self.convs[-1](x, edge_index)
        
        # with last linear layers
        x = self.linear_output(x)
        x = x.view(x.shape[0],self.out_channels,1)
        return x