import argparse
import hashlib
import os
import os.path as osp
import math
import glob
import json
import sqlite3

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from tqdm import tqdm, trange

from sing import (
    CostModel,
    random_share_assignment,
    parse_silph,
    share_types,
    random_valid_share_assignment,
    read_share_assignment_wires,
    save_share_assignment_text,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-dir",
    type=str,
    default="dataset/raw",
    help="Path to dataset directory",
)
parser.add_argument(
    "--mode",
    choices=["random", "all-b", "all-y", "perturb", "neighborloader", "op-type"],
    default="random",
    help="Which share assignments to generate",
)
parser.add_argument(
    "--noise-prob",
    type=float,
    default=0.5,
    help="random mode: Probability of picking a random share for each node",
)
parser.add_argument(
    "--base-assignment",
    choices=["silph", "all-b", "all-y"],
    default="silph",
    help="random, perturb, neighborloader, op-type modes: Base share assignment to modify",
)
parser.add_argument(
    "--perturb-n",
    type=int,
    default=1,
    help="perturb mode: Number of nodes to perturb",
)
parser.add_argument(
    "--perturb-stddev",
    type=int,
    default=3,
    help="perturb mode: Standard deviation around center of perturbation",
)
parser.add_argument(
    "--neighborloader-n",
    type=int,
    default=3,
    help="neighborloader mode: Number of nodes to perturb",
)
parser.add_argument(
    "--neighborloader-iterations",
    type=int,
    default=2,
    help="neighborloader mode: Number of hops to perform in neighbor sampling",
)
parser.add_argument(
    "--factor",
    type=float,
    default=3.0,
    help="Multiplication factor for number of samples to generate",
)
parser.add_argument(
    "--additional",
    action="store_true",
    help="Generate additional share assignments if some already exist",
)
parser.add_argument(
    "--only-valid",
    action="store_true",
    help="Only generate valid share assignments",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Don't save share assignments",
)
args = parser.parse_args()

splits = ["train", "val", "test"]

cost_model = CostModel("empirical_costs.json")

dataset_meta_path = osp.join(osp.dirname(__file__), "sing", "dataset.json")
with open(dataset_meta_path, "r") as f:
    dataset_meta = json.load(f)
hashes = sum([dataset_meta[split] for split in splits], [])

db_con = sqlite3.connect(osp.join(args.dataset_dir, "metadata.sqlite"))


def hash_tensor(tensor):
    h = hashlib.sha3_256()
    h.update(tensor.numpy().data)
    return h.hexdigest()

def save(circuit, share_assignment_wires, assignments_dir, shahash, y, y_hash, mode):
    if args.dry_run:
        print(shahash, y_hash)
    else:
        path = osp.join(assignments_dir, y_hash)
        path_txt = osp.join(assignments_dir, f"{y_hash}.txt")

        torch.save(y, path)
        save_share_assignment_text(y, share_assignment_wires, circuit, path_txt)

        metadata = [("mode", mode)]
        if mode == "random":
            metadata.append(("base-assignment", args.base_assignment))
            metadata.append(("noise-prob", f"{args.noise_prob}"))
        elif mode == "perturb":
            metadata.append(("base-assignment", args.base_assignment))
            metadata.append(("perturb-n", f"{args.perturb_n}"))
            metadata.append(("perturb-stddev", f"{args.perturb_stddev}"))
        elif mode == "neighborloader":
            metadata.append(("base-assignment", args.base_assignment))
            metadata.append(("neighborloader-n", f"{args.neighborloader_n}"))
            metadata.append(("neighborloader-iterations", f"{args.neighborloader_iterations}"))
        elif mode == "op-type":
            metadata.append(("base-assignment", args.base_assignment))

        data = [(shahash, y_hash, key, value) for key, value in metadata]
        with db_con:
            db_con.executemany("INSERT INTO metadata VALUES (?, ?, ?, ?)", data)


for shahash in tqdm(sorted(hashes)):
    bytecode_file = glob.glob(
        osp.join(args.dataset_dir, shahash, "*_c_main_bytecode.txt")
    )
    constant_input_file = glob.glob(
        osp.join(args.dataset_dir, shahash, "*_c_const.txt")
    )
    share_assignment_file = glob.glob(
        osp.join(args.dataset_dir, shahash, "*_c_share_map.txt")
    )

    assert len(bytecode_file) == 1
    assert len(constant_input_file) == 1
    assert len(share_assignment_file) == 1

    circuit, share_assignment, _ = parse_silph(
        bytecode_file[0],
        constant_input_file[0],
        share_assignment_file[0],
    )
    share_assignment_wires = read_share_assignment_wires(share_assignment_file[0])

    x = torch.tensor(circuit.nodes, dtype=torch.float)

    edge_index = torch.tensor(
        [circuit.edge_sources, circuit.edge_targets], dtype=torch.long
    )

    y = torch.tensor(share_assignment, dtype=torch.long)

    assignments_dir = osp.join(args.dataset_dir, shahash, "assignments")
    os.makedirs(assignments_dir, exist_ok=True)

    generated_hashes = os.listdir(assignments_dir)
    generated_hashes = list(filter(lambda x: len(x) == 64, generated_hashes))

    # "copy" silph share assignment
    y_hash = hash_tensor(y)
    if y_hash not in generated_hashes:
        save(circuit, share_assignment_wires, assignments_dir, shahash, y, y_hash, "silph")
        generated_hashes.append(y_hash)

    if len(generated_hashes) > 1 and not args.additional:
        continue

    all_b_assignment = torch.ones_like(y, dtype=torch.long)
    all_y_assignment = torch.ones_like(y, dtype=torch.long) * 2

    if args.mode in ["all-b", "all-y"]:
        if args.mode == "all-b":
            new_assignment = all_b_assignment
        elif args.mode == "all-y":
            new_assignment = all_y_assignment

        new_assignment_hash = hash_tensor(new_assignment)

        if new_assignment_hash not in generated_hashes:
            save(circuit, share_assignment_wires, assignments_dir, shahash, new_assignment, new_assignment_hash, args.mode)
            generated_hashes.append(new_assignment_hash)
    elif args.mode in ["random", "perturb", "neighborloader", "op-type"]:
        # generate noised/perturbed share assignments
        num_expected = max(3, math.ceil(args.factor * math.log2(len(circuit.nodes))))

        if args.base_assignment == "silph":
            base_assignment = y
        elif args.base_assignment == "all-b":
            base_assignment = all_b_assignment
        elif args.base_assignment == "all-y":
            base_assignment = all_y_assignment

        num_generate = num_expected
        for _ in range(num_generate):
            # create alternative_assignment
            if args.mode == "op-type":
                rand_idx = torch.randint(len(x), (1,)).item()
                op_type_one_hot = x[rand_idx]
                op_type = torch.argmax(op_type_one_hot)

                valid_share_types = 1 - cost_model.invalid[op_type]
                valid_share_types_dist = valid_share_types / torch.sum(valid_share_types)

                random_y_for_idx = torch.multinomial(valid_share_types_dist, num_samples=1).item()
                alternative_assignment = torch.ones_like(y) * random_y_for_idx
            else:
                if args.only_valid:
                    random_y = random_valid_share_assignment(x, cost_model)
                else:
                    random_y = random_share_assignment(x)

                alternative_assignment = random_y

            # create choices
            # 0: base, 1: alternative
            if args.mode == "random":
                dist = torch.distributions.bernoulli.Bernoulli(args.noise_prob)
                choices = dist.sample((len(x),))
            elif args.mode == "perturb":
                choices = torch.zeros((len(x),))

                center_idx = torch.randint(len(x), (1,)).item()
                stddev = args.perturb_stddev

                perturb_idx = torch.normal(center_idx, stddev, size=(args.perturb_n,)).round().long()
                perturb_idx = perturb_idx.clamp(0, len(x) - 1)

                choices[perturb_idx] = 1
            elif args.mode == "neighborloader":
                choices = torch.zeros((len(x),))

                num_neighbors = min(len(x), args.neighborloader_n)
                iterations = args.neighborloader_iterations
                data = Data(x=x, edge_index=edge_index)
                loader = NeighborLoader(data, num_neighbors=[num_neighbors] * iterations, shuffle=True)

                sampled_data = next(iter(loader))

                choices[sampled_data.n_id] = 1
            elif args.mode == "op-type":
                choices = torch.zeros((len(x),))
                choices[torch.argmax(x, dim=1) == op_type] = 1

            new_assignment = choices * alternative_assignment + (1 - choices) * base_assignment
            new_assignment = new_assignment.long()

            new_assignment_hash = hash_tensor(new_assignment)

            if new_assignment_hash not in generated_hashes:
                save(circuit, share_assignment_wires, assignments_dir, shahash, new_assignment, new_assignment_hash, args.mode)
                generated_hashes.append(new_assignment_hash)

db_con.close()
