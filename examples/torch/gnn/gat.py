# code modified from https://github.com/Diego999/pyGAT

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.linear_w = nn.Linear(in_features=in_features, out_features=out_features, bias=None)
        nn.init.xavier_uniform_(self.linear_w.weight.data, gain=1.414)

        self.linear_a_1 = nn.Linear(in_features=out_features, out_features=1, bias=None)
        self.linear_a_2 = nn.Linear(in_features=out_features, out_features=1, bias=None)
        nn.init.xavier_uniform_(self.linear_a_1.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.linear_a_2.weight.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout)

        self.register_buffer("zero_vec", None)

    def forward(self, h, adj):
        wh = self.linear_w(h)
        wh1 = self.linear_a_1(wh)
        wh2 = self.linear_a_2(wh)

        e = self.leakyrelu(wh1 + wh2.T)

        if self.zero_vec is None:
            self.zero_vec = -10e10 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, self.zero_vec)
        attention = self.dropout(self.softmax(attention))

        h_new = torch.matmul(attention, wh)

        if self.concat:
            return F.elu(h_new)
        else:
            return h_new


class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)
        ])

        self.last_layer = GATLayer(nhid * nheads,
                                   nclass,
                                   dropout=dropout,
                                   alpha=alpha,
                                   concat=False)

        self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)

    def forward(self, x, adj):
        x = self.dropout_1(x)

        multi_head_att = []
        for att in self.attentions:
            multi_head_att.append(att(x, adj))

        x = torch.cat(multi_head_att, dim=1)
        x = self.dropout_2(x)
        x = F.elu(self.last_layer(x, adj))

        return F.log_softmax(x, dim=1)
