import os.path as osp
import glob
import argparse
import time

import torch
import torch.nn.functional as F

from torcheval.metrics.functional import multiclass_confusion_matrix
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from sing import (
    ShareAssignmentModel,
    CostPredictionModel,
    CostModel,
    parse_silph,
    share_types,
    eval_cost,
    get_k,
    random_valid_share_assignment,
    read_share_assignment_wires,
    save_share_assignment_text,
    postprocess_share_assignment,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "circuit_dir",
    type=str,
    help="Path to directory containing circuit, constant input, and share assignment files",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoint.pt",
    help="Path to model checkpoint",
)
parser.add_argument(
    "--output",
    type=str,
    help="Path to output share assignment in Silph format",
)
parser.add_argument(
    "--calc-k", action="store_true", help="Calculate k for more accurate cost model"
)
parser.add_argument(
    "--cost-prediction",
    action="store_true",
    help="Also run cost prediction model and compare results",
)
args = parser.parse_args()

num_classes = len(share_types)

device = torch.device("cpu")

share_assignment_checkpoint = torch.load(args.checkpoint, map_location=device)

model = ShareAssignmentModel()
model.load_state_dict(share_assignment_checkpoint["state_dict"])
model.eval()

if args.cost_prediction:
    cost_prediction_checkpoint = torch.load("checkpoint-cost-prediction.pt")

    cost_prediction_model = CostPredictionModel()
    cost_prediction_model.load_state_dict(cost_prediction_checkpoint["state_dict"])
    cost_prediction_model.eval()

cost_model = CostModel("empirical_costs.json")

bytecode_file = glob.glob(osp.join(args.circuit_dir, "*_c_main_bytecode.txt"))
constant_input_file = glob.glob(osp.join(args.circuit_dir, "*_c_const.txt"))
share_assignment_file = glob.glob(osp.join(args.circuit_dir, "*_c_share_map.txt"))

assert len(bytecode_file) == 1
assert len(constant_input_file) == 1
assert len(share_assignment_file) == 1

circuit, share_assignment, _ = parse_silph(
    bytecode_file[0],
    constant_input_file[0],
    share_assignment_file[0],
)

data = Data(
    x=torch.tensor(circuit.nodes, dtype=torch.float),
    edge_index=torch.tensor(
        [circuit.edge_sources, circuit.edge_targets], dtype=torch.long
    ),
    y=torch.tensor(share_assignment, dtype=torch.long),
)

time_start = time.perf_counter_ns()
with torch.no_grad():
    out = model(
        x=data.x,
        edge_index=data.edge_index,
    )
time_end = time.perf_counter_ns()

if args.output:
    y = postprocess_share_assignment(data.x, out, cost_model)

    share_assignment_wires = read_share_assignment_wires(share_assignment_file[0])
    save_share_assignment_text(y, share_assignment_wires, circuit, args.output)

confusion_matrix = multiclass_confusion_matrix(out, data.y, num_classes)
num_nodes = data.num_nodes

assert torch.sum(confusion_matrix) == num_nodes

if args.calc_k:
    k = get_k(data.x, data.edge_index)
    cost_model.load_k(k)

random_y = random_valid_share_assignment(data.x, cost_model)

def run_cost_prediction_model(share_assignment_softmax):
    return cost_prediction_model(
        x=torch.cat((data.x, share_assignment_softmax), dim=1),
        edge_index=data.edge_index,
        batch=torch.zeros(data.num_nodes, dtype=torch.long),
    ).squeeze(0)


print(f"inference time [ms]: {(time_end - time_start) / (1000 * 1000)}")
print(confusion_matrix)
print(f"accuracy: {torch.sum(torch.diagonal(confusion_matrix)) / num_nodes:4f}")
print()

share_assignments = {
    "silph": F.one_hot(data.y, num_classes),
    "sing": out,
    "random": F.one_hot(random_y, num_classes),
    "all_b": F.one_hot(torch.ones_like(data.y), num_classes),
    "all_y": F.one_hot(torch.ones_like(data.y) * 2, num_classes),
}

for name, out_or_one_hot in share_assignments.items():
    print(name)

    invalid, cost = eval_cost(data.x, data.edge_index, torch.argmax(out_or_one_hot, dim=1), cost_model)
    print(f"invalid nodes: {invalid} cost: {cost}")

    if name in ["silph", "random"]:
        assert invalid == 0

    if args.cost_prediction:
        with torch.no_grad():
            cost_pred = run_cost_prediction_model(out_or_one_hot)

        print(f"predicted: invalid nodes: - cost: {cost_pred}")

    print()
