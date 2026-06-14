import os
import os.path as osp
import argparse
import uuid

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from torchmetrics.regression import (
    R2Score,
    MeanAbsoluteError,
    NormalizedRootMeanSquaredError,
)

import wandb

from sing import (
    CostPredictionSilphDataset,
    CostPredictionMeasuredDataset,
    CostPredictionModel,
    CostModel,
    assignment_filters,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs",
    type=int,
    default=150,
    help="Number of epochs",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    help="Learning rate",
)
parser.add_argument(
    "--mode",
    choices=["invalid", "cost", "combined"],
    default="cost",
    help="Train specific model",
)
parser.add_argument(
    "--invalid-cost",
    type=float,
    default=100,
    help="Cost assigned to invalid",
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
    "--cost-name",
    default="silph",
    help="Postfix of the text file containing the cost for each circuit",
)
parser.add_argument(
    "--assignment-filter",
    default="no_filter",
    help="Assignment filter to exclude specific assignments from the dataset",
)
parser.add_argument(
    "--load-checkpoint",
    type=str,
    help="Start training from a checkpoint",
)
parser.add_argument(
    "--name",
    type=str,
    help="wandb run name",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cost_model = CostModel("empirical_costs.json")


def load_dataset(split, force_reload=False):
    if args.cost_name == "silph":
        return CostPredictionSilphDataset(
            "dataset-cost-prediction",
            cost_model,
            split=split,
            force_reload=force_reload,
            only_valid=(args.mode == "cost"),
            assignment_filter=assignment_filters[args.assignment_filter],
        ).to(device)
    else:
        return CostPredictionMeasuredDataset(
            "dataset-cost-prediction-measured",
            split=split,
            force_reload=force_reload,
            only_valid=(args.mode == "cost"),
            cost_name=args.cost_name,
            assignment_filter=assignment_filters[args.assignment_filter],
        ).to(device)


train_dataset = load_dataset("train", True)
val_dataset = load_dataset("val")
test_dataset = load_dataset("test")

# intentionally after load_dataset as dataset loading is performed on
# the cpu
cost_model = cost_model.to(device)

for data in train_dataset:
    data.validate()
for data in val_dataset:
    data.validate()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


model = CostPredictionModel().to(device)

lr = args.lr
weight_decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

factor = 0.5
patience = 20
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

if args.mode in ["cost", "combined"]:
    loss_func = torch.nn.MSELoss()
elif args.mode == "invalid":
    loss_func = torch.nn.BCEWithLogitsLoss()

r2_score = R2Score().to(device)
mae = MeanAbsoluteError().to(device)
nrmse = NormalizedRootMeanSquaredError().to(device)

if args.load_checkpoint:
    cost_prediction_checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(cost_prediction_checkpoint["state_dict"])

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

    "scheduler": type(scheduler).__name__,
    "factor": factor,
    "patience": patience,

    "cost_name": args.cost_name,
    "assignment_filter": args.assignment_filter,
    "dataset_fingerprint": train_dataset.dataset_fingerprint,
    "dataset_args": train_dataset.dataset_meta["args"],
    "mode": args.mode,
    "train_size": len(train_dataset),
    "val_size": len(val_dataset),
    "test_size": len(test_dataset),
    "load_checkpoint": args.load_checkpoint,
}
if args.mode == "combined":
    wandb_config["invalid_cost"] = args.invalid_cost

run = wandb.init(
    entity=os.environ["WANDB_ENTITY"],
    project="sing",
    name=args.name,
    config=wandb_config,
    group="cost-prediction",
    tags=tags,
)
run.watch(model)


def get_y(dataset_y):
    cost = dataset_y[:, 1].unsqueeze(1)
    invalid = dataset_y[:, 0].unsqueeze(1)

    if args.mode == "cost":
        return cost
    elif args.mode == "invalid":
        return invalid
    elif args.mode == "combined":
        return invalid * args.invalid_cost + cost


def eval(loader):
    targets = []
    preds = []

    for batch in loader:
        with torch.no_grad():
            out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            y = get_y(batch.y)

            targets.append(y)
            preds.append(out)

    return targets, preds


def run_epoch(epoch):
    model.train()
    train_targets = []
    train_preds = []
    for batch in train_loader:
        optimizer.zero_grad()

        x = batch.x
        edge_index = batch.edge_index

        if args.delete_edges:
            edge_index = torch.tensor([[], []], dtype=torch.long).to(device)

        if args.zero_features:
            x[:] = 0

        out = model(x=x, edge_index=edge_index, batch=batch.batch)
        y = get_y(batch.y)
        loss = loss_func(out, y)

        loss.backward()
        optimizer.step()

        train_targets.append(y)
        train_preds.append(out)

    model.eval()
    val_targets, val_preds = eval(val_loader)
    test_targets, test_preds = eval(test_loader)

    train_targets = torch.cat(train_targets)
    train_preds = torch.cat(train_preds)
    val_targets = torch.cat(val_targets)
    val_preds = torch.cat(val_preds)
    test_targets = torch.cat(test_targets)
    test_preds = torch.cat(test_preds)

    train_loss = loss_func(train_preds, train_targets)
    val_loss = loss_func(val_preds, val_targets)

    if scheduler is not None:
        scheduler.step(val_loss)

    wandb_log = {
        "lr": optimizer.param_groups[0]["lr"],
        "Loss/train": train_loss,
        "Loss/val": val_loss,
        "Loss/test": loss_func(test_preds, test_targets),
    }

    if args.mode in ["cost", "combined"]:
        wandb_log.update({
            # train
            "R2/train": r2_score(train_preds, train_targets),
            "MAE/train": mae(train_preds, train_targets),
            "NRMSE/train": nrmse(train_preds, train_targets),
            # val
            "R2/val": r2_score(val_preds, val_targets),
            "MAE/val": mae(val_preds, val_targets),
            "NRMSE/val": nrmse(val_preds, val_targets),
            # test
            "R2/test": r2_score(test_preds, test_targets),
            "MAE/test": mae(test_preds, test_targets),
            "NRMSE/test": nrmse(test_preds, test_targets),
        })

        if epoch % 10 == 0:
            train_data = torch.cat((train_targets, train_preds), dim=1)
            val_data = torch.cat((val_targets, val_preds), dim=1)
            test_data = torch.cat((test_targets, test_preds), dim=1)

            train_table = wandb.Table(data=list(train_data), columns=["target", "pred"])
            val_table = wandb.Table(data=list(val_data), columns=["target", "pred"])
            test_table = wandb.Table(data=list(test_data), columns=["target", "pred"])

            wandb_log.update({
                "Scatter/train": wandb.plot.scatter(train_table, "target", "pred"),
                "Scatter/val": wandb.plot.scatter(val_table, "target", "pred"),
                "Scatter/test": wandb.plot.scatter(test_table, "target", "pred"),
            })
    elif args.mode == "invalid":
        wandb_log.update({
            "Accuracy/train": torch.sum(
                train_targets == (F.sigmoid(train_preds) > 0.5)
            )
            / len(train_dataset),
            "Accuracy/val": torch.sum(val_targets == (F.sigmoid(val_preds) > 0.5))
            / len(val_dataset),
        })

    run.log(wandb_log)

    print(f"epoch: {epoch} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")

    checkpoint = {
        "uuid": run_uuid,
        "cost_name": args.cost_name,
        "assignment_filter": args.assignment_filter,
        "dataset_args": train_dataset.dataset_meta["args"],

        "state_dict": model.state_dict(),

        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    checkpoint_dir = osp.join("checkpoints", run_uuid)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_file = osp.join(checkpoint_dir, f"epoch_{epoch:03}.pt")
    torch.save(checkpoint, checkpoint_file)

    checkpoint_symlink = "checkpoint-cost-prediction.pt"
    if osp.exists(checkpoint_symlink):
        os.remove(checkpoint_symlink)
    os.symlink(checkpoint_file, checkpoint_symlink)

for epoch in range(args.epochs):
    run_epoch(epoch)
