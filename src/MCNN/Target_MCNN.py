# Importing necessary libraries
import torch  # PyTorch for tensor operations and neural networks
import torch.nn as nn  # Neural network modules
from collections import OrderedDict  # OrderedDict for preserving order of elements in a dictionary

# Setting device to GPU if available, otherwise using CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Defining the TargetRepresentation class, which is a neural network module for target feature representation
class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()  # Calling the parent class constructor
        
        # Embedding layer to convert input tokens into dense vectors of size `embedding_num`
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        
        # Creating a list of convolutional blocks to process the embedded features
        self.block_list = nn.ModuleList()
        
        # Adding multiple convolutional blocks to the block list
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, embedding_num, 3)  # Convolution block with kernel size 3
            )
        
        # A linear layer to reduce the dimensionality after the convolutional blocks
        self.linear = nn.Linear(block_num * embedding_num, embedding_num)
        
    def forward(self, x):
        # Applying the embedding layer and changing the tensor's shape
        x = self.embed(x).permute(0, 2, 1)  # Permuting the tensor dimensions for convolution

        # Passing the input through each convolutional block and collecting the features
        feats = [block(x) for block in self.block_list]
        
        # Concatenating the features from all blocks along the last dimension
        x = torch.cat(feats, -1)
        
        # Passing the concatenated features through the linear layer
        x = self.linear(x)

        return x  # Returning the final processed features


# Defining the StackCNN class, which represents a stack of convolutional layers followed by pooling
class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()  # Calling the parent class constructor
        
        # Defining the sequence of layers: convolutional layers followed by batch normalization and ReLU activation
        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        
        # Adding additional convolutional layers if required
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        
        # Adding an adaptive max pooling layer to get a fixed-size output
        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        # Passing the input through the sequence of convolutional layers and pooling
        return self.inc(x).squeeze(-1)  # Removing the last dimension after pooling


# Defining the Conv1dReLU class, which represents a 1D convolution followed by batch normalization and ReLU activation
class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()  # Calling the parent class constructor
        
        # Defining a sequence of 1D convolution, batch normalization, and ReLU activation
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),  # Convolution layer
            nn.BatchNorm1d(out_channels),  # Batch normalization layer
            nn.ReLU()  # ReLU activation function
        )
    
    def forward(self, x):
        # Passing the input through the convolution, batch normalization, and ReLU activation layers
        return self.inc(x)
