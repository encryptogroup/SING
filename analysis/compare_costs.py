import argparse

import torch

from torcheval.metrics.functional import binary_confusion_matrix
from torchmetrics.regression import R2Score, PearsonCorrCoef

import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, callback, Output, Input

from sing import (
    CostPredictionSilphDataset,
    CostPredictionMeasuredDataset,
    CostModel,
    assignment_filters,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "cost_name_target",
    help="Target cost to compare with",
)
parser.add_argument(
    "cost_name_pred",
    help="Predicted cost to compare with",
)
parser.add_argument(
    "--assignment-filter",
    default="no_filter",
    help="Assignment filter to exclude specific assignments from the dataset",
)
parser.add_argument(
    "--dash",
    action="store_true",
    help="Start dash server for interactive plots",
)
args = parser.parse_args()

costs_target = {}
costs_pred = {}

mse_loss = torch.nn.MSELoss()
r2_score = R2Score()
pearson_corr_coef = PearsonCorrCoef()


def load_dataset(split, name, force_reload=False):
    if name in ["silph", "silph-wan"]:
        cost_model = CostModel("empirical_wan.json" if name == "silph-wan" else "empirical_costs.json")

        return CostPredictionSilphDataset(
            "dataset-cost-prediction",
            cost_model,
            split=split,
            force_reload=force_reload,
            only_valid=True,
            assignment_filter=assignment_filters[args.assignment_filter],
        )
    else:
        return CostPredictionMeasuredDataset(
            "dataset-cost-prediction-measured",
            split=split,
            force_reload=force_reload,
            only_valid=True,
            cost_name=name,
            assignment_filter=assignment_filters[args.assignment_filter],
        )


force_reload = True
for split in ["train", "val", "test"]:
    dataset_target = load_dataset(split, args.cost_name_target, force_reload)
    force_reload = False

    for data in dataset_target:
        if not data.shahash in costs_target:
            costs_target[data.shahash] = {}
        costs_target[data.shahash][data.assignment_hash] = data.y

force_reload = True
for split in ["train", "val", "test"]:
    dataset_pred = load_dataset(split, args.cost_name_pred, force_reload)
    force_reload = False

    for data in dataset_pred:
        if not data.shahash in costs_pred:
            costs_pred[data.shahash] = {}
        costs_pred[data.shahash][data.assignment_hash] = data.y

targets = []
preds = []

costs_columns = ["shahash", "assignment_hash", args.cost_name_target, args.cost_name_pred]
costs_table = []

for shahash, assignments in costs_target.items():
    for assignment_hash, y in assignments.items():
        if not shahash in costs_pred:
            continue

        if not assignment_hash in costs_pred[shahash]:
            continue

        target = y
        pred = costs_pred[shahash][assignment_hash]

        targets.append(target)
        preds.append(pred)

        # both valid
        if target[:, 0].item() == 0 and pred[:, 0].item() == 0:
            costs_table.append([
                shahash,
                assignment_hash,
                target[:, 1].item(),
                pred[:, 1].item(),
            ])

targets = torch.cat(targets)
preds = torch.cat(preds)

confusion_matrix = binary_confusion_matrix(preds[:, 0].long(), targets[:, 0].long())

# 0 where valid
both_valid = targets[:, 0] + preds[:, 0]
both_valid = both_valid == 0

mse = mse_loss(preds[:, 1][both_valid], targets[:, 1][both_valid])
r2 = r2_score(preds[:, 1][both_valid], targets[:, 1][both_valid])
pearson = pearson_corr_coef(preds[:, 1][both_valid], targets[:, 1][both_valid])

print(confusion_matrix)
print(f"mse: {mse}")
print(f"r2: {r2}")
print(f"pearson: {pearson}")

df = pd.DataFrame(costs_table, columns=costs_columns)

app = Dash()

shahashes_or_all = list(df.shahash.unique()) + ["!all"]
app.layout = [
    dcc.Dropdown(shahashes_or_all, "!all", id="dropdown-selection"),
    dcc.Graph(id="graph-content")
]

@callback(
    Output("graph-content", "figure"),
    Input("dropdown-selection", "value")
)
def update_graph(value):
    if value == "!all":
        dff = df
    else:
        dff = df[df.shahash==value]

    return px.scatter(dff, x=args.cost_name_target, y=args.cost_name_pred, hover_data=["shahash", "assignment_hash"])

if __name__ == "__main__" and args.dash:
    app.run(port=8053)
