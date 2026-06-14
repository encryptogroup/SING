import argparse

import torch
import torch.nn.functional as F

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input

from sing import (
    ops,
    share_types,
    CostPredictionSilphDataset,
    CostPredictionMeasuredDataset,
    CostModel,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--measured-dataset",
    action="store_true",
    help="Use measured dataset",
)
parser.add_argument(
    "--cost-name",
    default="cost",
    help="Postfix of the text file containing the cost for each circuit",
)
parser.add_argument(
    "--dash",
    action="store_true",
    help="Start dash server for interactive plots",
)
parser.add_argument(
    "--assignment-filter",
    default="no_filter",
    help="Assignment filter to exclude specific assignments from the dataset",
)
args = parser.parse_args()

num_ops = len(ops)
num_classes = len(share_types)

splits = ["train", "val", "test"]
validities = ["valid", "invalid"]

cost_model = CostModel("empirical_costs.json")

validity_for_split = {}
size_for_validity = {
    "valid": [],
    "invalid": [],
}

costs_columns = ["split", "shahash", "cost"]
costs_table = []


def load_dataset(split, force_reload=False):
    if args.measured_dataset:
        return CostPredictionMeasuredDataset(
            "dataset-cost-prediction-measured",
            split=split,
            force_reload=force_reload,
            only_valid=False,
            cost_name=args.cost_name,
            assignment_filter=assignment_filters[args.assignment_filter],
        )
    else:
        return CostPredictionSilphDataset(
            "dataset-cost-prediction",
            cost_model,
            split=split,
            force_reload=force_reload,
            only_valid=False,
            assignment_filter=assignment_filters[args.assignment_filter],
        )


force_reload = True
for split in splits:
    dataset = load_dataset(split, force_reload)
    force_reload = False

    validity_sum = torch.zeros(2)

    for data in dataset:
        validity = data.y[:, 0].squeeze(0).long()
        validity_one_hot = F.one_hot(validity, 2)

        validity_sum += validity_one_hot

        if validity == 0:
            # valid
            size_for_validity["valid"].append(data.num_nodes)

            costs_table.append([
                split,
                data.shahash,
                data.y[:, 1].squeeze(0),
            ])
        else:
            size_for_validity["invalid"].append(data.num_nodes)

    validity_for_split[split] = validity_sum


fig = go.Figure()
for validity in validities:
    fig.add_trace(
        go.Histogram(x=size_for_validity[validity], name=validity, bingroup=1)
    )

fig.update_layout(barmode="overlay")
fig.update_traces(opacity=0.75)
fig.write_html("validity_size_histogram.html")

data = pd.DataFrame(validity_for_split, index=validities)
fig = px.bar(data.T)
fig.write_html("validity.html")

df = pd.DataFrame(costs_table, columns=costs_columns)
fig = px.histogram(df, x="cost", color="split", marginal="rug")
fig.write_html("cost_histogram.html")


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

    return px.histogram(dff, x="cost", marginal="rug")

if __name__ == "__main__" and args.dash:
    app.run(port=8052)
