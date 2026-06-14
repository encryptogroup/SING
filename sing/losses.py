import torch
import torch.nn.functional as F


def invalid_assignment_loss(x, logits, cost_model):
    # for each node, which share assignment would be invalid?
    #
    # shape (num_nodes, num_classes)
    node_possible_invalid = torch.matmul(x, cost_model.invalid)

    y_softmax = torch.softmax(logits, dim=1)

    # for each node, 1 * logit if invalid assignment, 0 otherwise
    node_invalid = torch.einsum("ij,ij->i", node_possible_invalid, y_softmax)

    return torch.sum(node_invalid)


class CrossEntropyLoss(torch.nn.Module):
    """Identical behavior to torch.nn.CrossEntropyLoss, but with the
    same interface as the other losses we provide.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, batch, logits, target):
        # ignore x, edge_index, batch
        return F.cross_entropy(logits, target)


class InvalidAssignmentLoss(torch.nn.Module):
    def __init__(self, cost_model):
        super().__init__()
        self.cost_model = cost_model

    def forward(self, x, edge_index, batch, logits, target):
        # ignore edge_index, batch, target
        return invalid_assignment_loss(x, logits, self.cost_model)


class CombinedCrossEntropyInvalidAssignmentLoss(torch.nn.Module):
    def __init__(self, cost_model, alpha=0.5):
        super().__init__()
        self.cost_model = cost_model
        self.alpha = alpha

    def forward(self, x, edge_index, batch, logits, target):
        # ignore edge_index, batch
        cross_entropy = F.cross_entropy(logits, target)
        invalid_assignment = invalid_assignment_loss(x, logits, self.cost_model)

        return self.alpha * cross_entropy + (1 - self.alpha) * invalid_assignment


class CombinedPredictedCostInvalidAssignmentLoss(torch.nn.Module):
    def __init__(self, cost_prediction_model, cost_model, alpha=0.5):
        super().__init__()
        self.cost_prediction_model = cost_prediction_model
        self.cost_model = cost_model
        self.alpha = alpha

    def forward(self, x, edge_index, batch, logits, target):
        invalid_assignment = invalid_assignment_loss(x, logits, self.cost_model)

        x = torch.cat((x, F.softmax(logits, dim=1)), dim=1)

        prediction = self.cost_prediction_model(x=x, edge_index=edge_index, batch=batch)
        cost = torch.sum(prediction)

        return self.alpha * cost + (1 - self.alpha) * invalid_assignment


class PredictedCostLoss(torch.nn.Module):
    def __init__(self, cost_prediction_model):
        super().__init__()
        self.cost_prediction_model = cost_prediction_model

    def forward(self, x, edge_index, batch, logits, target):
        # ignore target
        x = torch.cat((x, F.softmax(logits, dim=1)), dim=1)

        prediction = self.cost_prediction_model(x=x, edge_index=edge_index, batch=batch)
        return torch.sum(prediction)
