import torch
import torch.nn.functional as F

from .circuit import share_types

num_classes = len(share_types)


def postprocess_share_assignment(x, logits, cost_model):
    y = torch.argmax(logits, dim=1)
    y_one_hot = F.one_hot(y, num_classes).float()

    # for each node, which share assignment would be invalid?
    #
    # shape (num_nodes, num_classes)
    node_possible_invalid = torch.matmul(x, cost_model.invalid)

    # for each node, 1 if invalid assignment, 0 otherwise
    node_invalid = torch.einsum("ij,ij->i", node_possible_invalid, y_one_hot)

    # fallback to yao which is always valid
    y[node_invalid > 0] = 2

    return y
