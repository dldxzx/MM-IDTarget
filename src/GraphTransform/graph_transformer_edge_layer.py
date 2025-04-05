import torch  # Import the PyTorch library for tensor operations and deep learning
import torch.nn as nn  # Import the neural network module from PyTorch, which contains pre-built layers and loss functions
import torch.nn.functional as F  # Import functional functions from PyTorch for various operations like activation functions
import dgl.function as fn  # Import DGL (Deep Graph Library) functions for graph-related operations
import numpy as np  # Import the NumPy library for numerical operations, particularly for arrays and matrix manipulation


# Function to compute dot product of source and destination fields for attention score
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}  # Calculate element-wise dot product between src and dst fields
    return func

# Function to scale the attention scores by a constant factor
def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}  # Divide attention score by a constant for scaling
    return func

# Function to adjust implicit attention scores with explicit edge features
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the implicit attention score (output of Q and K)
        explicit_edge: the explicit edge features to adjust the scores
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}  # Element-wise multiplication of implicit attention with explicit edge features
    return func

# Function to copy edge features to be passed to the feedforward network for edges
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}  # Copy edge features for use later
    return func


# Exponential function for numerical stability in softmax
def exp(field):
    def func(edges):
        # Apply clamp to the sum for numerical stability in softmax operation
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}  # Calculate exponential of the edge data with clamping for stability
    return func


"""
    Single Attention Head: Multi-Head Attention Layer
"""
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        # Initialize dimensions and number of heads
        self.out_dim = out_dim
        self.num_heads = num_heads

        # Define the linear layers for Q, K, V projections
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)  # Linear transformation for query
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)  # Linear transformation for key
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)  # Linear transformation for value
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)  # Linear transformation for edge features
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)  # No bias in transformation for query
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)  # No bias in transformation for key
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)  # No bias in transformation for value
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)  # No bias for edge feature transformation
    
    # Function to propagate attention scores through the graph
    def propagate_attention(self, g):
        # Compute attention score: dot product of K and Q
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # Apply dot product on edge features
        
        # Scaling the attention score by sqrt(out_dim) for numerical stability
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))  # Scale the scores by square root of the output dimension
        
        # Modify attention scores using explicit edge features if available
        g.apply_edges(imp_exp_attn('score', 'proj_e'))  # Adjust scores by multiplying with edge features
        
        # Copy the edge features for further processing in the next layer
        g.apply_edges(out_edge_features('score'))  # Copy the final edge features
        
        # Apply softmax on attention scores for normalization
        g.apply_edges(exp('score'))  # Exponential operation for numerical stability in softmax

        # Send weighted values from source to target nodes
        eids = g.edges()  # Get the edge IDs
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))  # Send and receive weighted values
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))  # Sum the edge scores for further use
    
    def forward(self, g, h, e):
        # Apply linear transformations to the input features for query, key, value, and edge features
        Q_h = self.Q(h)  # Apply Q transformation
        K_h = self.K(h)  # Apply K transformation
        V_h = self.V(h)  # Apply V transformation
        proj_e = self.proj_e(e)  # Apply edge feature transformation
        
        # Reshape data into [num_nodes, num_heads, feat_dim] for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)  # Reshape queries
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)  # Reshape keys
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)  # Reshape values
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)  # Reshape edge features
        
        # Propagate attention scores through the graph
        self.propagate_attention(g)
        
        # Calculate the final node features and edge features
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # Normalize by summing attention scores with added epsilon
        e_out = g.edata['e_out']  # Get the output edge features
        
        return h_out, e_out  # Return updated node and edge features
    

# Graph Transformer Layer incorporating Multi-Head Attention
class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, layer_norm=True, batch_norm=False, residual=True, use_bias=False):
        super().__init__()

        # Initialize parameters for the transformer layer
        self.in_channels = in_dim  # Input dimension for features
        self.out_channels = out_dim  # Output dimension for features
        self.num_heads = num_heads  # Number of attention heads
        self.dropout = dropout  # Dropout rate
        self.residual = residual  # Residual connections flag
        self.layer_norm = layer_norm  # Flag for applying layer normalization
        self.batch_norm = batch_norm  # Flag for applying batch normalization
        
        # Initialize Multi-Head Attention Layer
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)
        
        # Linear transformations for output features of nodes and edges
        self.O_h = nn.Linear(out_dim, out_dim)  # Output transformation for node features
        self.O_e = nn.Linear(out_dim, out_dim)  # Output transformation for edge features

        # Layer normalization if specified
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)  # Layer norm for node features
            self.layer_norm1_e = nn.LayerNorm(out_dim)  # Layer norm for edge features
            
        # Batch normalization if specified
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)  # Batch norm for node features
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)  # Batch norm for edge features
        
        # Feed-Forward Network (FFN) for nodes
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)  # First layer of FFN for nodes
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)  # Second layer of FFN for nodes
        
        # Feed-Forward Network (FFN) for edges
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)  # First layer of FFN for edges
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)  # Second layer of FFN for edges

        # Layer normalization for FFN output
        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)  # Layer norm for node FFN output
            self.layer_norm2_e = nn.LayerNorm(out_dim)  # Layer norm for edge FFN output
            
        # Batch normalization for FFN output
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)  # Batch norm for node FFN output
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)  # Batch norm for edge FFN output
        
    def forward(self, g, h, e):
        h_in1 = h  # Save input node features for first residual connection
        e_in1 = e  # Save input edge features for first residual connection
        
        # Multi-head attention output
        h_attn_out, e_attn_out = self.attention(g, h, e)
        
        h = h_attn_out.view(-1, self.out_channels)  # Flatten node features
        e = e_attn_out.view(-1, self.out_channels)  # Flatten edge features
        
        h = F.dropout(h, self.dropout, training=self.training)  # Apply dropout on node features
        e = F.dropout(e, self.dropout, training=self.training)  # Apply dropout on edge features

        # Linear transformations for node and edge features
        h = self.O_h(h)  # Apply output transformation for nodes
        e = self.O_e(e)  # Apply output transformation for edges

        if self.residual:
            h = h_in1 + h  # Add residual connection for nodes
            e = e_in1 + e  # Add residual connection for edges

        # Apply layer normalization if specified
        if self.layer_norm:
            h = self.layer_norm1_h(h)  # Apply layer norm for nodes
            e = self.layer_norm1_e(e)  # Apply layer norm for edges

        # Apply batch normalization if specified
        if self.batch_norm:
            h = self.batch_norm1_h(h)  # Apply batch norm for nodes
            e = self.batch_norm1_e(e)  # Apply batch norm for edges

        h_in2 = h  # Save node features for second residual connection
        e_in2 = e  # Save edge features for second residual connection

        # Feedforward Network for nodes
        h = self.FFN_h_layer1(h)
        h = F.relu(h)  # Apply ReLU activation
        h = F.dropout(h, self.dropout, training=self.training)  # Apply dropout
        h = self.FFN_h_layer2(h)

        # Feedforward Network for edges
        e = self.FFN_e_layer1(e)
        e = F.relu(e)  # Apply ReLU activation
        e = F.dropout(e, self.dropout, training=self.training)  # Apply dropout
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # Add second residual connection for nodes       
            e = e_in2 + e  # Add second residual connection for edges  

        # Apply layer normalization for the output if specified
        if self.layer_norm:
            h = self.layer_norm2_h(h)  # Apply layer norm for nodes after FFN
            e = self.layer_norm2_e(e)  # Apply layer norm for edges after FFN

        # Apply batch normalization for the output if specified
        if self.batch_norm:
            h = self.batch_norm2_h(h)  # Apply batch norm for nodes after FFN
            e = self.batch_norm2_e(e)  # Apply batch norm for edges after FFN             

        return h, e  # Return the final node and edge features
        
    def __repr__(self):
        # Return a string representation of the layer
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
