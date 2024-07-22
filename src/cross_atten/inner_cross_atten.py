import torch
import torch.nn as nn
import numpy as np

#Transformer Parameters
d_model = 128 #Embedding Size
d_ff = 512 #FeedForward dimension
d_k = d_v = 16 #dimension of K(=Q), V
n_layers = 1 #number of Encoder
n_heads = 8 #number of heads in Multi-Head Attention

class Inner_EncoderLayer(nn.Module):
    def __init__(self):
        super(Inner_EncoderLayer, self).__init__()
        self.enc_self_attn = Inner_MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, inputs_Q, inputs_K, inputs_V, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(inputs_Q, inputs_K, inputs_V, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) 
        return enc_outputs, attn
    
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to('cuda:0')(output + residual) 


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) 
        return context, attn


class Inner_MultiHeadAttention(nn.Module):
    def __init__(self):
        super(Inner_MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size, len_q = input_Q, input_Q.size(0), input_Q.size(1)
        input_K1 = input_K.unsqueeze(1).repeat(1, len_q, 1)
        input_V1 = input_V.unsqueeze(1).repeat(1, len_q, 1)
        Q = self.W_Q(input_Q.float()).view(batch_size, -1, n_heads, d_k).transpose(1, 2) 
        K = self.W_K(input_K1).view(batch_size, -1, n_heads, d_k).transpose(1, 2) 
        V = self.W_V(input_V1).view(batch_size, -1, n_heads, d_v).transpose(1, 2) 
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads*d_v) 
        output = self.fc(context) 
        return nn.LayerNorm(d_model).to('cuda:0')(output + residual), attn
    
