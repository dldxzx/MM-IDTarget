# Import necessary libraries
import torch  # PyTorch library for tensor operations and neural network components
import torch.nn as nn  # PyTorch's neural network module for defining layers and models
import numpy as np  # Numpy library for numerical operations

# Transformer Parameters
d_model = 128  # Size of the embedding (the dimensionality of input and output vectors)
d_ff = 512  # Dimension of the FeedForward network's hidden layer
d_k = d_v = 16  # Dimension of Key (K) and Value (V) vectors (in multi-head attention)
n_layers = 1  # Number of layers in the Encoder
n_heads = 8  # Number of attention heads in the multi-head attention mechanism

# Defining the Inner_EncoderLayer class
class Inner_EncoderLayer(nn.Module):
    def __init__(self):
        super(Inner_EncoderLayer, self).__init__()  # Initialize the parent class
        self.enc_self_attn = Inner_MultiHeadAttention()  # Multi-head attention mechanism
        self.pos_ffn = PoswiseFeedForwardNet()  # Position-wise feed-forward network

    def forward(self, inputs_Q, inputs_K, inputs_V, enc_self_attn_mask):
        # Pass the inputs through the multi-head attention and position-wise feed-forward network
        enc_outputs, attn = self.enc_self_attn(inputs_Q, inputs_K, inputs_V, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # Apply the position-wise feed-forward network
        return enc_outputs, attn  # Return the output and attention weights
    
    
# Defining the position-wise feed-forward network (FFN) class
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()  # Initialize the parent class
        self.fc = nn.Sequential(  # Define a sequence of fully connected layers
            nn.Linear(d_model, d_ff, bias=False),  # First linear layer from d_model to d_ff
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Linear(d_ff, d_model, bias=False)  # Second linear layer from d_ff back to d_model
        )

    def forward(self, inputs):
        residual = inputs  # Save the input for residual connection
        output = self.fc(inputs)  # Pass the inputs through the feed-forward network
        # Apply layer normalization and add the residual connection before returning the result
        return nn.LayerNorm(d_model).to('cuda:0')(output + residual) 


# Scaled Dot-Product Attention mechanism
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()  # Initialize the parent class

    def forward(self, Q, K, V, attn_mask):
        # Compute the dot product between Query and Key, then scale by the square root of d_k
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        if attn_mask is not None:
            # Apply the attention mask (if provided) to prevent attending to certain positions
            scores.masked_fill_(attn_mask, -1e9) 
        # Apply softmax to get attention weights
        attn = nn.Softmax(dim=-1)(scores)
        # Compute the context by multiplying the attention weights with the Values
        context = torch.matmul(attn, V) 
        return context, attn  # Return the context and attention weights


# Inner Multi-Head Attention mechanism class
class Inner_MultiHeadAttention(nn.Module):
    def __init__(self):
        super(Inner_MultiHeadAttention, self).__init__()  # Initialize the parent class
        # Linear transformations for Query, Key, and Value (no bias term)
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention()  # Scaled Dot-Product Attention
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)  # Final linear layer to combine outputs

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size, len_q = input_Q, input_Q.size(0), input_Q.size(1)  # Get input dimensions
        input_K1 = input_K.unsqueeze(1).repeat(1, len_q, 1)  # Expand Key tensor to match Query sequence length
        input_V1 = input_V.unsqueeze(1).repeat(1, len_q, 1)  # Expand Value tensor similarly

        # Apply linear transformations to Query, Key, and Value, and reshape them for multi-head attention
        Q = self.W_Q(input_Q.float()).view(batch_size, -1, n_heads, d_k).transpose(1, 2) 
        K = self.W_K(input_K1).view(batch_size, -1, n_heads, d_k).transpose(1, 2) 
        V = self.W_V(input_V1).view(batch_size, -1, n_heads, d_v).transpose(1, 2) 

        # Pass the transformed Q, K, and V through Scaled Dot-Product Attention
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)

        # Reshape and apply the final linear layer
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads*d_v) 
        output = self.fc(context)  # Final output after linear transformation

        # Apply layer normalization and return the output with the residual connection
        return nn.LayerNorm(d_model).to('cuda:0')(output + residual), attn  # Return output and attention weights
