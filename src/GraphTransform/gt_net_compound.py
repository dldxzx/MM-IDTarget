import torch.nn as nn  # Importing PyTorch's neural network module
from GraphTransform.graph_transformer_edge_layer import GraphTransformerLayer  # Import the GraphTransformerLayer class from another module

# Define the GraphTransformer class, which inherits from nn.Module
class GraphTransformer(nn.Module):
    # Initialize the GraphTransformer with various parameters such as device, dimensions, and hyperparameters
    def __init__(self, device, n_layers, node_dim, edge_dim, hidden_dim, out_dim, n_heads, in_feat_dropout, dropout, pos_enc_dim):
        super(GraphTransformer, self).__init__()  # Initialize the parent class nn.Module

        # Device configuration: specifies whether the model will run on GPU or CPU
        self.device = device

        # Boolean flags for different configurations
        self.layer_norm = True  # Whether to use layer normalization
        self.batch_norm = False  # Whether to use batch normalization (set to False)
        self.residual = True  # Whether to use residual connections

        # Linear layer for transforming node features to hidden dimensions
        self.linear_h = nn.Linear(node_dim, hidden_dim)

        # Linear layer for transforming edge features to hidden dimensions
        self.linear_e = nn.Linear(edge_dim, hidden_dim)

        # Dropout layer for the input features
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # Linear layer for embedding Laplacian positional encoding into hidden dimensions
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        # Create a list of GraphTransformerLayer for each layer, except the last one
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, self.layer_norm,
                                                           self.batch_norm, self.residual)
                                     for _ in range(n_layers - 1)])

        # Add the last GraphTransformerLayer with the output dimension
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, n_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))

        # Batch normalization layers for each transformer layer
        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])  # Add batch normalization for each layer

    # The forward function, where the input graph data is passed through the network
    def forward(self, g):
        # Move the graph to the specified device (GPU/CPU)
        g = g.to(self.device)

        # Extract node features (atoms) and move them to the specified device, convert to float
        h = g.ndata['atom'].float().to(self.device)

        # Extract Laplacian positional encoding features and move to the specified device
        h_lap_pos_enc = g.ndata['lap_pos_enc'].to(self.device)

        # Extract edge features (bonds) and move them to the specified device, convert to float
        e = g.edata['bond'].float().to(self.device)

        # Pass node features through a linear layer
        h = self.linear_h(h)

        # Embed Laplacian positional encoding through a linear layer and convert to float
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())

        # Add Laplacian positional encoding to the node features
        h = h + h_lap_pos_enc

        # Apply dropout to the node features
        h = self.in_feat_dropout(h)

        # Pass edge features through the edge linear layer
        e = self.linear_e(e)

        # Apply each GraphTransformerLayer and batch normalization after each layer
        for conv, bn in zip(self.layers, self.batch_norm_layers):  # Loop through layers and batch normalization
            h, e = conv(g, h, e)  # Apply the graph transformer layer
            h = bn(h)  # Apply batch normalization to the node features

        # Store the final node features in the graph data dictionary
        g.ndata['h'] = h

        # Return the final node features after processing
        return h
