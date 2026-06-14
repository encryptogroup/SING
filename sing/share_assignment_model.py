import torch
from torch_geometric.nn import SAGEConv, DirGNNConv

from .circuit import ops, share_types

node_feature_size = len(ops)
num_classes = len(share_types)


class ShareAssignmentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_channels = 64

        self.conv1 = SAGEConv(node_feature_size, self.hidden_channels)
        self.conv1 = DirGNNConv(self.conv1)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.act1 = torch.nn.ReLU()

        self.conv2 = SAGEConv(self.hidden_channels, num_classes)
        self.conv2 = DirGNNConv(self.conv2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)

        return x

    def __str__(self):
        return f"{type(self.conv1).__name__}-{self.hidden_channels}"
