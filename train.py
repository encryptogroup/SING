import os
import os.path as osp
import argparse
import uuid

import torch
import torch.nn.functional as F

from torcheval.metrics.functional import multiclass_confusion_matrix

from torch_geometric.loader import DataLoader

import wandb

from sing import (
    SilphDataset,
    ShareAssignmentModel,
    CostPredictionModel,
    CostModel,
    eval_cost,
    CombinedCrossEntropyInvalidAssignmentLoss,
    CombinedPredictedCostInvalidAssignmentLoss,
    PredictedCostLoss,
    share_types,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    help="Learning rate",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="Blending between cross-entropy loss and invalid assignment loss",
)
parser.add_argument(
    "--predicted-cost",
    action="store_true",
    help="Use trained CostPredictionModel",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoint-cost-prediction.pt",
    help="Path to model checkpoint",
)
parser.add_argument(
    "--delete-edges",
    action="store_true",
    help="Ablation: remove all edges during training",
)
parser.add_argument(
    "--zero-features",
    action="store_true",
    help="Ablation: zero out node features during training",
)
parser.add_argument(
    "--name",
    type=str,
    help="wandb run name",
)
args = parser.parse_args()

num_classes = len(share_types)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cost_model = CostModel("empirical_costs.json").to(device)

train_dataset = SilphDataset("dataset", split="train", force_reload=True).to(device)
val_dataset = SilphDataset("dataset", split="val").to(device)
test_dataset = SilphDataset("dataset", split="test").to(device)

train_num_nodes = 0
val_num_nodes = 0
test_num_nodes = 0
for data in train_dataset:
    data.validate()
    train_num_nodes += data.num_nodes
for data in val_dataset:
    data.validate()
    val_num_nodes += data.num_nodes
for data in test_dataset:
    data.validate()
    test_num_nodes += data.num_nodes

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

if args.predicted_cost:
    cost_prediction_checkpoint = torch.load(args.checkpoint)

    cost_prediction_model = CostPredictionModel()
    cost_prediction_model.load_state_dict(cost_prediction_checkpoint["state_dict"])
    cost_prediction_model = cost_prediction_model.to(device)
    cost_prediction_model.eval()


model = ShareAssignmentModel().to(device)
lr = args.lr
weight_decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
alpha = args.alpha

if args.predicted_cost:
    loss_func = CombinedPredictedCostInvalidAssignmentLoss(
        cost_prediction_model, cost_model, alpha=alpha
    )
    # loss_func = PredictedCostLoss(cost_prediction_model)
else:
    loss_func = CombinedCrossEntropyInvalidAssignmentLoss(cost_model, alpha=alpha)

run_uuid = str(uuid.uuid4())

tags = []

if args.delete_edges:
    tags.append("delete-edges")
if args.zero_features:
    tags.append("zero-features")

wandb_config = {
    "run_uuid": run_uuid,
    "model": f"{model}",
    "loss": type(loss_func).__name__,
    "optimizer": type(optimizer).__name__,
    "lr": lr,
    "weight_decay": weight_decay,
    "alpha": alpha,
    "dataset_fingerprint": train_dataset.dataset_fingerprint,
    "dataset_args": train_dataset.dataset_meta["args"],
    "train_size": len(train_dataset),
    "val_size": len(val_dataset),
    "test_size": len(test_dataset),
}
if args.predicted_cost:
    wandb_config["cost_name"] = cost_prediction_checkpoint["cost_name"]
    wandb_config["cost_model_uuid"] = cost_prediction_checkpoint["uuid"]

run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY", ""),
    project="sing",
    name=args.name,
    config=wandb_config,
    group="share-assignment",
    tags=tags,
)
run.watch(model)


def wandb_plot_cm(confusion_matrix, title):
    class_names = share_types
    data = [
        [class_names[i], class_names[j], confusion_matrix[i, j]]
        for i in range(num_classes)
        for j in range(num_classes)
    ]

    return wandb.plot.custom_chart.plot_table(
        data_table=wandb.Table(
            columns=["Actual", "Predicted", "nPredictions"],
            data=data,
        ),
        vega_spec_name="wandb/confusion_matrix/v1",
        fields={
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        },
        string_fields={"title": title},
        split_table=False,
    )


def eval(loader, cost_model):
    total_loss = 0.0
    targets = []
    preds = []
    invalids = []
    costs = []

    for batch in loader:
        with torch.no_grad():
            out = model(x=batch.x, edge_index=batch.edge_index)
            loss = loss_func(batch.x, batch.edge_index, batch.batch, out, batch.y)

            y = torch.argmax(out, dim=1)
            invalid, cost = eval_cost(batch.x, batch.edge_index, y, cost_model)

        total_loss += loss * len(batch)
        targets.append(batch.y)
        preds.append(y)
        invalids.append(invalid)
        costs.append(cost)

    return total_loss, targets, preds, invalids, costs


def run_epoch(epoch):
    model.train()
    train_loss = 0.0
    train_preds = []
    train_invalid = 0
    train_cost = 0
    train_correct = 0
    train_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long).to(device)
    for batch in train_loader:
        optimizer.zero_grad()

        x = batch.x
        edge_index = batch.edge_index

        if args.delete_edges:
            edge_index = torch.tensor([[], []], dtype=torch.long)

        if args.zero_features:
            x[:] = 0

        out = model(x=x, edge_index=edge_index)
        loss = loss_func(x, edge_index, batch.batch, out, batch.y)

        loss.backward()
        optimizer.step()

        y = torch.argmax(out, dim=1)
        invalid, cost = eval_cost(batch.x, batch.edge_index, y, cost_model)

        train_loss += loss * len(batch) / len(train_dataset)
        train_preds.append(y)
        train_invalid += invalid
        train_cost += cost
        train_correct += torch.sum(y == batch.y)
        train_confusion_matrix += multiclass_confusion_matrix(out, batch.y, num_classes)

    model.eval()
    val_loss, val_targets, val_preds, val_invalids, val_costs = eval(
        val_loader, cost_model
    )
    test_loss, test_targets, test_preds, test_invalids, test_costs = eval(
        test_loader, cost_model
    )

    val_loss = val_loss / len(val_dataset)
    test_loss = test_loss / len(test_dataset)

    train_preds = torch.cat(train_preds)

    val_targets = torch.cat(val_targets)
    val_preds = torch.cat(val_preds)
    test_targets = torch.cat(test_targets)
    test_preds = torch.cat(test_preds)

    val_invalid = sum(val_invalids)
    val_cost = sum(val_costs)
    val_correct = torch.sum(val_targets == val_preds)
    val_confusion_matrix = multiclass_confusion_matrix(
        val_preds, val_targets, num_classes
    )

    test_invalid = sum(test_invalids)
    test_cost = sum(test_costs)
    test_correct = torch.sum(test_targets == test_preds)
    test_confusion_matrix = multiclass_confusion_matrix(
        test_preds, test_targets, num_classes
    )

    train_share_dist = torch.sum(F.one_hot(train_preds, num_classes), dim=0)
    val_share_dist = torch.sum(F.one_hot(val_preds, num_classes), dim=0)
    test_share_dist = torch.sum(F.one_hot(test_preds, num_classes), dim=0)

    run.log({
        "lr": optimizer.param_groups[0]["lr"],

        "Loss/train": train_loss,
        "Invalid/train": train_invalid,
        "Cost/train": train_cost,
        "Accuracy/train": train_correct / train_num_nodes,
        "arith/train": train_share_dist[share_types.index("a")] / train_num_nodes,
        "bool/train": train_share_dist[share_types.index("b")] / train_num_nodes,
        "yao/train": train_share_dist[share_types.index("y")] / train_num_nodes,
        # "Confusion Matrix/train": wandb_plot_cm(train_confusion_matrix, "train"),

        "Loss/val": val_loss,
        "Invalid/val": val_invalid,
        "Cost/val": val_cost,
        "Accuracy/val": val_correct / val_num_nodes,
        "arith/val": val_share_dist[share_types.index("a")] / val_num_nodes,
        "bool/val": val_share_dist[share_types.index("b")] / val_num_nodes,
        "yao/val": val_share_dist[share_types.index("y")] / val_num_nodes,
        # "Confusion Matrix/val": wandb_plot_cm(val_confusion_matrix, "val"),

        "Loss/test": test_loss,
        "Invalid/test": test_invalid,
        "Cost/test": test_cost,
        "Accuracy/test": test_correct / test_num_nodes,
        "arith/test": test_share_dist[share_types.index("a")] / test_num_nodes,
        "bool/test": test_share_dist[share_types.index("b")] / test_num_nodes,
        "yao/test": test_share_dist[share_types.index("y")] / test_num_nodes,
        # "Confusion Matrix/test": wandb_plot_cm(test_confusion_matrix, "test"),
    })

    print(f"epoch: {epoch} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")

    checkpoint = {
        "uuid": run_uuid,
        "dataset_args": train_dataset.dataset_meta["args"],

        "state_dict": model.state_dict(),

        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if args.predicted_cost:
        checkpoint["cost_name"] = cost_prediction_checkpoint["cost_name"]

    checkpoint_dir = osp.join("checkpoints", run_uuid)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_file = osp.join(checkpoint_dir, f"epoch_{epoch:03}.pt")
    torch.save(checkpoint, checkpoint_file)

    checkpoint_symlink = "checkpoint.pt"
    if osp.exists(checkpoint_symlink):
        os.remove(checkpoint_symlink)
    os.symlink(checkpoint_file, checkpoint_symlink)

for epoch in range(200):
    run_epoch(epoch)
