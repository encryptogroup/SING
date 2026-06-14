import torch
import torch.nn.functional as F

from .circuit import share_types

num_classes = len(share_types)


def eval_cost(x, edge_index, y, cost_model):
    y_one_hot = F.one_hot(y, num_classes).float()

    # for each node, which share assignment would be invalid?
    #
    # shape (num_nodes, num_classes)
    node_possible_invalid = torch.matmul(x, cost_model.invalid)

    # for each node, 1 if invalid assignment, 0 otherwise
    node_invalid = torch.einsum("ij,ij->i", node_possible_invalid, y_one_hot)

    # TODO in smart mode, scale cost of select and store with array width
    node_possible_cost = torch.matmul(x, cost_model.op_cost)
    node_cost = torch.einsum("ij,ij->i", node_possible_cost, y_one_hot)

    edge_from_y = y[edge_index[0]]
    edge_to_y = y[edge_index[1]]

    edge_cost = cost_model.conv_cost[edge_from_y, edge_to_y]

    return torch.sum(node_invalid), torch.sum(node_cost) + torch.sum(edge_cost)
