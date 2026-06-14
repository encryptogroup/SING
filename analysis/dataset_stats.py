import torch
import torch.nn.functional as F

from statistics import mean, stdev

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sing import ops, share_types, SilphDataset

num_ops = len(ops)
num_classes = len(share_types)

splits = ["train", "val", "test"]

num_nodes_for_split = {}
share_types_for_split = {}
ops_for_split = {}

force_reload = True
for split in splits:
    dataset = SilphDataset("dataset-cost-prediction", split=split, force_reload=force_reload)
    force_reload = False

    num_nodes = []
    share_types_sum = torch.zeros(num_classes)
    ops_sum = torch.zeros(num_ops)

    for data in dataset:
        y_one_hot = F.one_hot(data.y, num_classes)

        num_nodes.append(data.num_nodes)
        share_types_sum += torch.sum(y_one_hot, dim=0)
        ops_sum += torch.sum(data.x, dim=0)

    num_nodes_for_split[split] = num_nodes
    share_types_for_split[split] = share_types_sum
    ops_for_split[split] = ops_sum

num_nodes = sum(num_nodes_for_split.values(), [])

print("circuits", sum(len(x) for x in num_nodes_for_split.values()))
print("min", min(num_nodes))
print("max", max(num_nodes))
print("mean", mean(num_nodes))
print("stdev", stdev(num_nodes))

fig = go.Figure()
for split in splits:
    fig.add_trace(go.Histogram(x=num_nodes_for_split[split], name=split, bingroup=1))

# Overlay both histograms
fig.update_layout(barmode="overlay")
# Reduce opacity to see all histograms
fig.update_traces(opacity=0.75)
fig.write_html("histogram.html")

data = pd.DataFrame(share_types_for_split, index=share_types)
fig_share_types = px.bar(data.T)
fig_share_types.write_html("share_types.html")

data = pd.DataFrame(ops_for_split, index=ops)
fig_ops = px.bar(data.T)
fig_ops.write_html("ops.html")
