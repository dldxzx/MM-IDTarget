import torch
import torch.nn as nn
import numpy as np

# Transformer parameters
d_model = 128  # Embedding size (dimensionality of model representation)
d_ff = 512     # Feed-forward network dimension
d_k = d_v = 16 # Dimensions of Q (Query) and K (Key), V (Value) in Attention mechanism
n_layers = 1   # Number of Encoder layers
n_heads = 8    # Number of attention heads in Multi-Head Attention


class Inter_EncoderLayer(nn.Module):
    def __init__(self):
        super(Inter_EncoderLayer, self).__init__()
        # Initialize Multi-Head Attention and Feed-Forward Network
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, inputs_Q, inputs_K, inputs_V, enc_self_attn_mask):
        # Perform self-attention and pass the results through the feed-forward network
        enc_outputs, attn = self.enc_self_attn(inputs_Q, inputs_K, inputs_V, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # Apply the position-wise feed-forward network
        return enc_outputs, attn  # Return outputs and attention weights


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # Define a simple two-layer feed-forward network
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),  # First Linear Layer: d_model -> d_ff
            nn.ReLU(inplace=True),                 # ReLU activation
            nn.Linear(d_ff, d_model, bias=False)   # Second Linear Layer: d_ff -> d_model
        )

    def forward(self, inputs):
        # Apply the feed-forward network and add the residual connection
        residual = inputs  # Save the input for residual connection
        output = self.fc(inputs)  # Pass input through the feed-forward network
        return nn.LayerNorm(d_model).to('cuda:0')(output + residual)  # Apply LayerNorm and return


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Compute the attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # Scaled dot-product of Q and K
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Apply attention mask (set masked positions to a very low value)
        attn = nn.Softmax(dim=-1)(scores)  # Apply softmax to get attention weights
        context = torch.matmul(attn, V)  # Compute context vector by weighting V with attention weights
        return context, attn  # Return context and attention weights


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # Initialize the layers for multi-head attention
        self.fc0 = nn.Linear(d_model, d_model, bias=False)  # Linear layer for residual connection
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # Projection for Query
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)  # Projection for Key
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)  # Projection for Value
        self.ScaledDotProductAttention = ScaledDotProductAttention()  # Scaled Dot-Product Attention
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)  # Linear layer for output of multi-head attention

    def forward(self, input_Q, input_K, input_V, attn_mask):
        if attn_mask is not None:
            # If an attention mask is provided, apply it to the input (residual connection)
            batch_size, seq_len, model_len = input_Q.size()  # Get dimensions of input
            residual_2D = input_Q.view(batch_size * seq_len, model_len)  # Reshape for linear projection
            residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)  # Apply linear transformation
        else:
            residual, batch_size = input_Q, input_Q.size(0)  # Otherwise, use the input directly for residual

        # Apply the projections for Q, K, V
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Project Q
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Project K
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # Project V

        if attn_mask is not None:
            # Repeat attention mask for each head
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # Apply scaled dot-product attention
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)
        
        # Reshape the context back and apply the final linear layer
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # Reshape context to match output
        output = self.fc(context)  # Apply linear transformation to combine the results of all heads

        # Apply Layer Normalization and add residual connection before returning
        return nn.LayerNorm(d_model).to('cuda:0')(output + residual), attn  # Return output and attention weights
