import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

import argparse

from sing import (
    SilphDataset,
    Circuit,
    CostModel,
    ops,
    encode_node_features,
    ShareAssignmentModel,
    CostPredictionModel,
    eval_cost,
    get_k,
)

parser = argparse.ArgumentParser()
parser.add_argument("gate_type", type=str, choices=ops)
parser.add_argument(
    "--calc-k", action="store_true", help="Calculate k for more accurate cost model"
)
args = parser.parse_args()

share_assignment_checkpoint = torch.load("checkpoint.pt")

model = ShareAssignmentModel()
model.load_state_dict(share_assignment_checkpoint["state_dict"])
model.eval()

cost_prediction_checkpoint = torch.load("checkpoint-cost-prediction.pt")

cost_prediction_model = CostPredictionModel()
cost_prediction_model.load_state_dict(cost_prediction_checkpoint["state_dict"])
cost_prediction_model.eval()

cost_model = CostModel("empirical_costs.json")

x = torch.tensor(
    [
        encode_node_features["IN"],
        encode_node_features["IN"],
        encode_node_features[args.gate_type],
    ],
    dtype=torch.float,
)
edge_index = torch.tensor(
    [
        [0, 2],
        [1, 2],
    ],
    dtype=torch.long,
).t()

with torch.no_grad():
    out = model(x=x, edge_index=edge_index)

if args.calc_k:
    k = get_k(x, edge_index)
    cost_model.load_k(k)

y = torch.argmax(out, dim=1)
invalid, cost = eval_cost(x, edge_index, y, cost_model)

with torch.no_grad():
    cost_pred = cost_prediction_model(
        x=torch.cat((x, F.softmax(out, dim=1)), dim=1),
        edge_index=edge_index,
        batch=torch.zeros(len(x), dtype=torch.long),
    ).squeeze(0)


print(f"out: {out}")
print(f"y: {y}")
print(f"invalid nodes: {invalid} cost: {cost}")
print(f"predicted: invalid nodes: - cost: {cost_pred}")
