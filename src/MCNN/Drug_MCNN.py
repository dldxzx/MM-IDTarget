# Importing necessary libraries
import torch  # PyTorch for tensor operations and neural networks
import torch.nn as nn  # Neural network modules
from collections import OrderedDict  # OrderedDict for preserving order of elements in a dictionary

# Setting the device to GPU if available, otherwise using CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Defining the DrugRepresentation class, which is a neural network for drug feature representation
class DrugRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        # Embedding layer to convert drug indices into embedding vectors
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        
        # List of convolutional blocks to process the embedded features
        self.block_list = nn.ModuleList()
        
        # Creating multiple convolutional blocks
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, embedding_num, 3)  # 3 is the kernel size
            )
        
        # Linear layer to reduce the dimensionality after the convolutional blocks
        self.linear = nn.Linear(block_num * embedding_num, embedding_num)
        
    def forward(self, x):
        # Convert drug indices into embeddings and permute the dimensions
        x = self.embed(x).permute(0, 2, 1)
        
        # Pass through each convolutional block and collect the features
        feats = [block(x) for block in self.block_list]
        
        # Concatenate all the features along the last dimension (features from all blocks)
        x = torch.cat(feats, -1)
        
        # Pass the concatenated features through the linear layer
        x = self.linear(x)

        return x 


# Defining the StackCNN class, which is a stack of 1D convolutional layers with ReLU activation
class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        # Initializing the first convolutional layer
        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        
        # Adding subsequent convolutional layers if more than one layer is needed
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        
        # Adding an adaptive max pooling layer that outputs a fixed-size output
        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        # Passing the input through the stack of convolutional layers and pooling
        return self.inc(x).squeeze(-1)  # Removing the last dimension after pooling


# Defining the Conv1dReLU class, which represents a 1D convolution layer followed by Batch Normalization and ReLU activation
class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Defining a sequence of convolution, batch normalization, and ReLU activation
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),  # Batch Normalization to stabilize training
            nn.ReLU()  # ReLU activation function for non-linearity
        )
    
    def forward(self, x):
        # Passing the input through the convolutional block
        return self.inc(x)
