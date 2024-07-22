import torch
import torch.nn as nn
from GraphTransform import gt_net_compound
from MCNN import Target_MCNN
from MCNN import Drug_MCNN
from EW_GCN import EW_GCN
from cross_atten import inner_cross_atten
from cross_atten import inter_cross_atten
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 128 #Embedding Size

class MM_IDTarget(nn.Module):
    def __init__(self, block_num, vocab_protein_size, embedding_size=128, in_dim = 33):
        super(MM_IDTarget,self).__init__()
        self.tgcn = EW_GCN.GCN(in_dim, hidden_dim = 128, out_dim = 128)
        self.compound_gt = gt_net_compound.GraphTransformer(device, n_layers=3, node_dim=44, edge_dim=10, hidden_dim=128,
                                                        out_dim=128, n_heads=8, in_feat_dropout=0.0, dropout=0.1, pos_enc_dim=8)
        self.smile_encoder = Drug_MCNN.DrugRepresentation(block_num, 77, 128)
        self.protein_encoder = Target_MCNN.TargetRepresentation(block_num, vocab_protein_size, 128)
        #share weights
        self.inner_cross_atten = inner_cross_atten.Inner_EncoderLayer()
        self.inter_cross_atten = inter_cross_atten.Inter_EncoderLayer()
        self.projection = nn.Sequential(
            nn.Linear(d_model*2, d_model*4),  # 减少隐藏层维度
            nn.LayerNorm(d_model*4),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(d_model*4, d_model),  # 将隐藏层维度减少到与输入维度相等
        )
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(2042,1024) 
        self.linear1 = nn.Linear(1024,256)
        self.linear2 = nn.Linear(256,64)
        self.linear3 = nn.Linear(64,2)
        self.relu = nn.ReLU()
        self.softmax = nn.Sigmoid()

    
    def dgl_split(self, bg, feats):
        max_num_nodes = int(bg.batch_num_nodes().max())
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],
                       dim=1).reshape(-1).type(torch.long).to(bg.device)
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
        return out
    
    
    def forward(self, mol_graph, pro_graph, pro_A, target, smile, ecfp4): 
        pdb_graph = self.tgcn(pro_graph, pro_graph.ndata['x'], pro_graph.edata['w'])
        compound_graphtransformer = self.compound_gt(mol_graph)
        compound_feat_mol_graph = self.dgl_split(mol_graph, compound_graphtransformer)
        # 将张量进行最大池化
        pooling = nn.AdaptiveMaxPool1d(1)
        smile_emb = pooling(compound_feat_mol_graph.permute(0, 2, 1)).squeeze(2)
        protein_x = self.protein_encoder(target)
        smiles_x = self.smile_encoder(smile)
        #inner_cross_atten based on graph information for drug and target
        # 定义线性变换层
        InnerAtten_outD, _ = self.inner_cross_atten(smiles_x, smile_emb, smile_emb, None)
        InnerAtten_outT, _ = self.inner_cross_atten(protein_x, pdb_graph, pdb_graph, None)
        #inter_cross_atten for drug and target
        T2D_out, _ = self.inter_cross_atten(InnerAtten_outD, InnerAtten_outT, InnerAtten_outT, None)
        D2T_out, _ = self.inter_cross_atten(InnerAtten_outT, InnerAtten_outD, InnerAtten_outD, None)
        #seq features plus graph features
        din = torch.cat((torch.sum(T2D_out, 1), smile_emb), 1)
        tin = torch.cat((torch.sum(D2T_out, 1), pdb_graph), 1)
        #linear projection for drug and target
        dout = self.projection(din)
        tout = self.projection(tin)
        # exit()
        x = torch.cat((dout, tout, ecfp4, pro_A), 1)
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
        x = self.softmax(x)
        return x
    

    