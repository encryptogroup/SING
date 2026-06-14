import argparse
from operator import itemgetter

import torch

from torchmetrics.regression import R2Score, PearsonCorrCoef

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table

from sing import (
    CostPredictionSilphDataset,
    CostPredictionMeasuredDataset,
    CostPredictionModel,
    CostModel,
    assignment_filters,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoint-cost-prediction.pt",
    help="Path to model checkpoint",
)
parser.add_argument(
    "--mode",
    choices=["invalid", "cost", "combined"],
    default="cost",
    help="Mode in which model was trained in",
)
parser.add_argument(
    "--cost-name",
    help="Postfix of the text file containing the cost for each circuit",
)
parser.add_argument(
    "--dash",
    action="store_true",
    help="Start dash server for interactive plots",
)
parser.add_argument(
    "--assignment-filter",
    help="Assignment filter to exclude specific assignments from the dataset",
)
args = parser.parse_args()

cost_model = CostModel("empirical_costs.json")

cost_prediction_checkpoint = torch.load(args.checkpoint)
run_uuid = cost_prediction_checkpoint["uuid"]
train_cost_name = cost_prediction_checkpoint["cost_name"]
train_assignment_filter = cost_prediction_checkpoint["assignment_filter"]
dataset_args = cost_prediction_checkpoint["dataset_args"]

model = CostPredictionModel()
model.load_state_dict(cost_prediction_checkpoint["state_dict"])
model.eval()

cost_name = args.cost_name
if cost_name is None:
    cost_name = train_cost_name
    print(f"inferred cost_name {cost_name} from checkpoint")

assignment_filter = args.assignment_filter
if assignment_filter is None:
    assignment_filter = train_assignment_filter
    print(f"inferred assignment_filter {assignment_filter} from checkpoint")

mse_loss = torch.nn.MSELoss()
r2_score = R2Score()
pearson_corr_coef = PearsonCorrCoef()


def load_dataset(split, force_reload=False):
    if cost_name == "silph":
        return CostPredictionSilphDataset(
            "dataset-cost-prediction",
            cost_model,
            split=split,
            force_reload=force_reload,
            only_valid=(args.mode == "cost"),
            assignment_filter=assignment_filters[assignment_filter],
        )
    else:
        return CostPredictionMeasuredDataset(
            "dataset-cost-prediction-measured",
            split=split,
            force_reload=force_reload,
            only_valid=(args.mode == "cost"),
            cost_name=cost_name,
            assignment_filter=assignment_filters[assignment_filter],
        )


def get_y(dataset_y):
    cost = dataset_y[:, 1].unsqueeze(1)
    invalid = dataset_y[:, 0].unsqueeze(1)

    if args.mode == "cost":
        return cost
    elif args.mode == "invalid":
        return invalid
    elif args.mode == "combined":
        return invalid * args.invalid_cost + cost


cost_predictions_columns = ["split", "shahash", "assignment_hash", cost_name, "predicted", "mode", "base_assignment"]
cost_predictions_table = []


force_reload = True
for split in ["train", "val", "test"]:
    dataset = load_dataset(split, force_reload)
    force_reload = False

    print(f"split: {split} len: {len(dataset)}")

    targets = []
    pred = []
    num_nodes = []

    for data in dataset:
        with torch.no_grad():
            out = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
            y = get_y(data.y)

        targets.append(y)
        pred.append(out)
        num_nodes.append(torch.tensor([[data.num_nodes]]))

        cost_predictions_table.append([
            split,
            data.shahash,
            data.assignment_hash,
            y.item(),
            out.item(),
            data.mode,
            data["base-assignment"],
        ])

    targets = torch.cat(targets)
    pred = torch.cat(pred)
    num_nodes = torch.cat(num_nodes).float()

    mse = mse_loss(pred, targets)
    r2 = r2_score(pred, targets)
    pearson = pearson_corr_coef(pred, targets)

    mse_num_nodes = mse_loss(num_nodes, targets)
    r2_num_nodes = r2_score(num_nodes, targets)
    pearson_num_nodes = pearson_corr_coef(num_nodes, targets)

    print(f"mse: {mse}")
    print(f"r2: {r2}")
    print(f"pearson: {pearson}")
    print()
    print(f"mse (num_nodes): {mse_num_nodes}")
    print(f"r2 (num_nodes): {r2_num_nodes}")
    print(f"pearson (num_nodes): {pearson_num_nodes}")
    print()

df = pd.DataFrame(cost_predictions_table, columns=cost_predictions_columns)

app = Dash()

dropdown_options = ["!all"] + [f"!{mode}" for mode in df["mode"].unique()] + list(df.shahash.unique())
app.layout = [
    html.H1("Cost Prediction Evaluation"),
    html.H2("Model information"),
    html.P(f"UUID: {run_uuid}"),
    html.P(f"trained with cost_name {train_cost_name} and assignment_filter {train_assignment_filter}"),
    html.P("dataset_args:"),
    dash_table.DataTable([{"key": key, "value": value} for key, value in dataset_args.items()]),
    html.H2("Scatterplot"),
    html.Div([
        # next and prev button at the same time don't work for some reason
        # html.Button("<", id="button-prev", n_clicks=0),
        html.Button(">", id="button-next", n_clicks=0),
    ]),
    dcc.Dropdown([{"label": label, "value": idx} for idx, label in enumerate(dropdown_options)], 0, id="dropdown-selection"),
    dcc.Graph(id="graph-content"),
]


# @callback(
#     Output("dropdown-selection", "value"),
#     Input("button-prev", "n_clicks"),
#     State("dropdown-selection", "value"),
#     prevent_initial_call=True
# )
# def prev_dropdown(n_clicks, value):
#     return (value - 1) % len(dropdown_options)

@callback(
    Output("dropdown-selection", "value"),
    Input("button-next", "n_clicks"),
    State("dropdown-selection", "value"),
    prevent_initial_call=True
)
def next_dropdown(n_clicks, value):
    return (value + 1) % len(dropdown_options)

@callback(
    Output("graph-content", "figure"),
    Input("dropdown-selection", "value")
)
def update_graph(value):
    value = dropdown_options[value]

    if value == "!all":
        dff = df
    elif value.startswith("!"):
        dff = df[df["mode"] == value[1:]]
    else:
        dff = df[df.shahash == value]

    fig = px.scatter(dff, x=cost_name, y="predicted", color="mode", hover_data=["shahash", "assignment_hash", "base_assignment"])

    min_x = min(dff[cost_name].min(), dff["predicted"].min())
    max_x = max(dff[cost_name].max(), dff["predicted"].max())
    x = [min_x, max_x]
    fig.add_trace(go.Scatter(x=x, y=x, mode="lines", name="perfect prediction"))

    return fig

if __name__ == "__main__" and args.dash:
    app.run(port=8051)
