# Importing the necessary modules from PyTorch for neural network building and manipulation
import torch
import torch.nn as nn

# Importing a graph transformer module for processing compound graphs
from GraphTransform import gt_net_compound

# Importing the MCNN (Multi-scale Convolutional Neural Network) module for protein and drug representation
from MCNN import Target_MCNN
from MCNN import Drug_MCNN

# Importing a graph convolutional network (GCN) module for graph-based learning
from EW_GCN import EW_GCN

# Importing inner and inter cross-attention modules for attention-based learning between drug and target
from cross_atten import inner_cross_atten
from cross_atten import inter_cross_atten


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding size (dimension of the model)
d_model = 128

class MM_IDTarget(nn.Module):
    def __init__(self, block_num, vocab_protein_size, embedding_size=128, in_dim = 33):
        # Initialize the MM_IDTarget model
        super(MM_IDTarget,self).__init__()

        # Define the graph convolutional network (GCN) for the protein graph
        self.tgcn = EW_GCN.GCN(in_dim, hidden_dim = 128, out_dim = 128)

        # Define the graph transformer for compound graph
        self.compound_gt = gt_net_compound.GraphTransformer(device, n_layers=3, node_dim=44, edge_dim=10, hidden_dim=128,
                                                        out_dim=128, n_heads=8, in_feat_dropout=0.0, dropout=0.1, pos_enc_dim=8)

        # Define the drug representation encoder using MCNN
        self.smile_encoder = Drug_MCNN.DrugRepresentation(block_num, 77, 128)

        # Define the target (protein) representation encoder using MCNN
        self.protein_encoder = Target_MCNN.TargetRepresentation(block_num, vocab_protein_size, 128)

        # Define inner cross-attention layer to process drug and target features
        self.inner_cross_atten = inner_cross_atten.Inner_EncoderLayer()

        # Define inter cross-attention layer for interactions between drug and target features
        self.inter_cross_atten = inter_cross_atten.Inter_EncoderLayer()

        # Define a projection layer to project the concatenated drug and target features
        self.projection = nn.Sequential(
            nn.Linear(d_model*2, d_model*4),  # Reduce hidden layer dimension
            nn.LayerNorm(d_model*4),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(d_model*4, d_model),  # Reduce the hidden layer dimension to match input dimension
        )

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.1)

        # Define fully connected layers for the final prediction
        self.linear = nn.Linear(2042, 1024)
        self.linear1 = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 2)

        # ReLU activation function
        self.relu = nn.ReLU()

        # Sigmoid activation for binary classification output
        self.softmax = nn.Sigmoid()

    def dgl_split(self, bg, feats):
        # This function splits the graph data and applies padding to the feature tensor
        max_num_nodes = int(bg.batch_num_nodes().max())  # Get the maximum number of nodes in the graph
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],
                       dim=1).reshape(-1).type(torch.long).to(bg.device)  # Batch indices for each node
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])  # Cumulative number of nodes
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)  # Reindex nodes in the batch
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]  # Define the output size
        out = feats.new_full(size, fill_value=0)  # Initialize the output tensor with zeros
        out[idx] = feats  # Assign the features to the correct index
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])  # Reshape the output
        return out

    def forward(self, mol_graph, pro_graph, pro_A, target, smile, ecfp4):
        # Forward pass of the model

        # Apply the graph convolutional network (GCN) on the protein graph
        pdb_graph = self.tgcn(pro_graph, pro_graph.ndata['x'], pro_graph.edata['w'])

        # Apply the compound graph transformer on the molecular graph
        compound_graphtransformer = self.compound_gt(mol_graph)

        # Transform the compound features
        compound_feat_mol_graph = self.dgl_split(mol_graph, compound_graphtransformer)

        # Apply max pooling on the transformed molecular features
        pooling = nn.AdaptiveMaxPool1d(1)
        smile_emb = pooling(compound_feat_mol_graph.permute(0, 2, 1)).squeeze()

        # Encode the protein and drug (smiles) representations
        protein_x = self.protein_encoder(target)
        smiles_x = self.smile_encoder(smile)

        # Inner cross-attention based on drug and target graph information
        InnerAtten_outD, _ = self.inner_cross_atten(smiles_x, smile_emb, smile_emb, None)
        InnerAtten_outT, _ = self.inner_cross_atten(protein_x, pdb_graph, pdb_graph, None)

        # Inter cross-attention for interactions between drug and target
        T2D_out, _ = self.inter_cross_atten(InnerAtten_outD, InnerAtten_outT, InnerAtten_outT, None)
        D2T_out, _ = self.inter_cross_atten(InnerAtten_outT, InnerAtten_outD, InnerAtten_outD, None)

        # Concatenate features from drug and target interactions with graph features
        din = torch.cat((torch.sum(T2D_out, 1), smile_emb), 1)
        tin = torch.cat((torch.sum(D2T_out, 1), pdb_graph), 1)

        # Apply linear projection to reduce the dimensionality for drug and target features
        dout = self.projection(din)
        tout = self.projection(tin)

        # Concatenate the projected drug and target features with additional features (ecfp4 and pro_A)
        x = torch.cat((dout, tout, ecfp4, pro_A), 1)

        # Pass through fully connected layers with ReLU activations and dropout for regularization
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)

        # Apply Sigmoid for binary classification output
        x = self.softmax(x)
        return x
