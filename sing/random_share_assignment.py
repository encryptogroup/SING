import torch

from .circuit import ops, share_types

node_feature_size = len(ops)
num_classes = len(share_types)


def random_share_assignment(x):
    shape = (len(x),)
    return torch.randint(num_classes, shape)


def random_valid_share_assignment(x, cost_model):
    valid = 1 - cost_model.invalid

    prob_dist = valid / torch.sum(valid, dim=1).unsqueeze(1)

    node_op = torch.argmax(x, dim=1)
    node_prob_dist = prob_dist[node_op]

    return torch.multinomial(node_prob_dist, num_samples=1).squeeze(1)
