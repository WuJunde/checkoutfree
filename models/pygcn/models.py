import torch.nn as nn
import torch.nn.functional as F
from models.pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout = None):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        if self.dropout:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
