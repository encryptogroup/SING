import torch
from torch_geometric.nn import SAGEConv, pool, MLP

from .circuit import ops, share_types

node_feature_size = len(ops)
num_classes = len(share_types)


class CostPredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_channels = 16

        self.conv1 = SAGEConv(node_feature_size + num_classes, self.hidden_channels)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.act1 = torch.nn.ReLU()

        self.conv2 = SAGEConv(self.hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        cost = self.conv1(x, edge_index)
        cost = self.act1(cost)
        cost = self.dropout1(cost)

        cost = self.conv2(cost, edge_index)

        cost = pool.global_add_pool(cost, batch)

        return cost

    def __str__(self):
        return f"{type(self.conv1).__name__}-{self.hidden_channels}"
