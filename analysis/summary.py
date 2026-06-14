import torch

from sing import ops, share_types, ShareAssignmentModel, CostPredictionModel

from torch_geometric.nn import summary

num_nodes = 100
x = torch.zeros(num_nodes, len(ops))
edge_index = torch.randint(num_nodes, size=(2, 20))
batch = torch.cat(
    (
        torch.zeros(num_nodes // 2, dtype=torch.long),
        torch.ones(num_nodes - num_nodes // 2, dtype=torch.long),
    )
)  # two equally sized batches
out = torch.zeros(num_nodes, len(share_types))

print(summary(ShareAssignmentModel(), x, edge_index))

print(summary(CostPredictionModel(), torch.cat((x, out), 1), edge_index, batch))
