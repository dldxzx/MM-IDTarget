import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.layer1 = dglnn.GraphConv(in_dim, hidden_dim*4, bias=False)
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*4, bias=False),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(inplace=True),
        )
        self.layer2 = dglnn.GraphConv(hidden_dim*8, hidden_dim*4, bias=False)
        self.layer3 = dglnn.GraphConv(hidden_dim*4, out_dim, bias=False)

    def forward(self, graph, x, w):
        w, _ = torch.max(w, dim=1)
        x1 = self.layer1(graph, x, edge_weight=w)
        x1 = F.relu(x1, inplace=True)
        f1 = self.fc1(x)
        x1f1 = torch.cat((x1, f1), 1)
        x2 = self.layer2(graph, x1f1, edge_weight=w)
        x2 = F.relu(x2, inplace=True)
        x3 = self.layer3(graph, x2, edge_weight=w)
        x3 = F.relu(x3, inplace=True)
        
        with graph.local_scope():
            graph.ndata['x'] = x3
            readout = dgl.sum_nodes(graph, 'x')
            # 归一化
            readout = F.normalize(readout, p=2, dim=1)  
            return readout
        