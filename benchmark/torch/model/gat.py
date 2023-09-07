# code modified from https://github.com/Diego999/pyGAT

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear_w = nn.Linear(in_features=in_features, out_features=out_features, bias=None)

        self.linear_a_1 = nn.Linear(in_features=out_features, out_features=1, bias=None)
        self.linear_a_2 = nn.Linear(in_features=out_features, out_features=1, bias=None)

        self.leakyrelu = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h, adj):
        wh = self.linear_w(h)
        wh1 = self.linear_a_1(wh)
        wh2 = self.linear_a_2(wh)

        e = self.leakyrelu(wh1 + wh2.T)
        zero_vec = -10e10 * torch.ones(*e.shape, device=e.device, dtype=e.dtype)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = self.softmax(attention)

        h_new = torch.matmul(attention, wh)

        return F.elu(h_new)
