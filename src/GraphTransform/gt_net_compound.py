import torch.nn as nn
from GraphTransform.graph_transformer_edge_layer import GraphTransformerLayer


class GraphTransformer(nn.Module):
    def __init__(self, device, n_layers, node_dim, edge_dim, hidden_dim, out_dim, n_heads, in_feat_dropout, dropout, pos_enc_dim):
        super(GraphTransformer, self).__init__()

        self.device = device
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.linear_h = nn.Linear(node_dim, hidden_dim)
        self.linear_e = nn.Linear(edge_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, self.layer_norm,
                                                           self.batch_norm, self.residual)
                                     for _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, n_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])  # 添加批量归一化层

    def forward(self, g):
        # input embedding
        g = g.to(self.device)
        h = g.ndata['atom'].float().to(self.device)
        h_lap_pos_enc = g.ndata['lap_pos_enc'].to(self.device)
        e = g.edata['bond'].float().to(self.device)

        h = self.linear_h(h)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        h = h + h_lap_pos_enc
        h = self.in_feat_dropout(h)

        e = self.linear_e(e)

        for conv, bn in zip(self.layers, self.batch_norm_layers):  # 在每个图变换层后添加归一化
            h, e = conv(g, h, e)
            h = bn(h)
        # for conv in self.layers:
        #     h, e = conv(g, h, e)

        g.ndata['h'] = h

        return h
