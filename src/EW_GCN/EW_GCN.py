# Importing PyTorch library for tensor operations and deep learning
import torch

# Importing the neural network modules from PyTorch
import torch.nn as nn

# Importing DGL (Deep Graph Library) for graph-based neural network operations
import dgl.nn.pytorch as dglnn

# Importing DGL (Deep Graph Library) for graph operations
import dgl

# Importing functional operations from PyTorch (like activation functions, etc.)
import torch.nn.functional as F


# Defining a Graph Convolutional Network (GCN) class
class GCN(nn.Module):
    # Initializing the GCN model with input, hidden, and output dimensions
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        
        # Defining the first graph convolutional layer (GraphConv) with input and hidden dimensions, no bias
        self.layer1 = dglnn.GraphConv(in_dim, hidden_dim*4, bias=False)
        
        # Defining a fully connected layer followed by layer normalization and ReLU activation
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*4, bias=False),  # Linear layer
            nn.LayerNorm(hidden_dim*4),  # Layer normalization for stabilizing the training
            nn.ReLU(inplace=True),  # ReLU activation function for non-linearity
        )
        
        # Defining the second graph convolutional layer with hidden dimensions
        self.layer2 = dglnn.GraphConv(hidden_dim*8, hidden_dim*4, bias=False)
        
        # Defining the third graph convolutional layer with output dimension
        self.layer3 = dglnn.GraphConv(hidden_dim*4, out_dim, bias=False)

    # Defining the forward pass of the GCN model
    def forward(self, graph, x, w):
        # Reducing the edge weights `w` by taking the maximum along the second dimension
        w, _ = torch.max(w, dim=1)
        
        # Passing the input through the first graph convolutional layer and applying ReLU activation
        x1 = self.layer1(graph, x, edge_weight=w)
        x1 = F.relu(x1, inplace=True)
        
        # Passing the input through the fully connected layer with normalization and activation
        f1 = self.fc1(x)
        
        # Concatenating the output of the first graph convolution and the fully connected layer
        x1f1 = torch.cat((x1, f1), 1)
        
        # Passing the concatenated features through the second graph convolutional layer and applying ReLU
        x2 = self.layer2(graph, x1f1, edge_weight=w)
        x2 = F.relu(x2, inplace=True)
        
        # Passing the output through the third graph convolutional layer and applying ReLU
        x3 = self.layer3(graph, x2, edge_weight=w)
        x3 = F.relu(x3, inplace=True)
        
        # Using local scope in DGL to prevent accidental overwriting of graph data
        with graph.local_scope():
            # Storing the output of the third graph convolutional layer in the graph node data
            graph.ndata['x'] = x3
            
            # Performing readout (aggregation) by summing the node features
            readout = dgl.sum_nodes(graph, 'x')
            
            # Normalizing the readout across nodes (L2 normalization)
            readout = F.normalize(readout, p=2, dim=1)
            
            # Returning the normalized readout feature vector
            return readout
