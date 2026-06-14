import argparse

import torch
import torch.nn.functional as F

from torcheval.metrics.functional import multiclass_confusion_matrix

import pandas as pd

from sing import (
    SilphDataset,
    ShareAssignmentModel,
    CostModel,
    share_types,
    ops,
    eval_cost,
    get_k,
    random_valid_share_assignment,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoint.pt",
    help="Path to model checkpoint",
)
parser.add_argument(
    "--calc-k", action="store_true", help="Calculate k for more accurate cost model"
)
args = parser.parse_args()

num_ops = len(ops)
num_classes = len(share_types)

share_assignment_checkpoint = torch.load(args.checkpoint)

model = ShareAssignmentModel()
model.load_state_dict(share_assignment_checkpoint["state_dict"])
model.eval()

cost_model = CostModel("empirical_costs.json")

for split in ["train", "val", "test"]:
    dataset = SilphDataset("dataset", split=split)
    print(f"split: {split}")

    op_share_assignment = torch.zeros((num_ops, num_classes))

    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    num_nodes = 0

    invalid_nodes = 0
    invalid_circuits = 0
    costs_silph = []
    costs_random = []
    costs = []

    for data in dataset:
        with torch.no_grad():
            out = model(x=data.x, edge_index=data.edge_index)

        confusion_matrix += multiclass_confusion_matrix(out, data.y, num_classes)
        num_nodes += data.num_nodes

        if args.calc_k:
            k = get_k(data.x, data.edge_index)
            cost_model.load_k(k)

        random_y = random_valid_share_assignment(data.x, cost_model)

        y = torch.argmax(out, dim=1)
        y_one_hot = F.one_hot(y, num_classes)

        invalid_silph, cost_silph = eval_cost(
            data.x, data.edge_index, data.y, cost_model
        )
        invalid, cost = eval_cost(data.x, data.edge_index, y, cost_model)
        invalid_random, cost_random = eval_cost(
            data.x, data.edge_index, random_y, cost_model
        )

        assert invalid_silph == 0
        assert invalid_random == 0

        if invalid > 0:
            invalid_nodes += invalid
            invalid_circuits += 1
        else:
            costs_silph.append(cost_silph)
            costs.append(cost)
            costs_random.append(cost_random)

        op_share_assignments_one_hot = torch.matmul(data.x.unsqueeze(2), y_one_hot.float().unsqueeze(1))
        op_share_assignment += torch.sum(op_share_assignments_one_hot, dim=0)

    assert torch.sum(confusion_matrix) == num_nodes

    op_share_assignment_table = pd.DataFrame(op_share_assignment.long(), index=ops, columns=share_types)

    print(confusion_matrix)
    print(f"accuracy: {torch.sum(torch.diagonal(confusion_matrix)) / num_nodes:4f}")
    print(f"invalid: {invalid_nodes} nodes in {invalid_circuits} circuits")
    print()
    print(
        f"avg abs cost diff: {torch.mean(torch.tensor(costs) - torch.tensor(costs_silph))}"
    )
    print(
        f"avg rel cost diff: {torch.mean(torch.tensor(costs) / torch.tensor(costs_silph))}"
    )
    print()
    print("random assignment:")
    print(
        f"avg abs cost diff: {torch.mean(torch.tensor(costs_random) - torch.tensor(costs_silph))}"
    )
    print(
        f"avg rel cost diff: {torch.mean(torch.tensor(costs_random) / torch.tensor(costs_silph))}"
    )
    print()
    print(f"op -> share assignment dist:")
    print(op_share_assignment_table)
    print()
