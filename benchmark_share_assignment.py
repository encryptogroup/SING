import os
import os.path as osp

import argparse
import subprocess
import glob
import json
import tempfile
import time
import hashlib
import csv
from statistics import mean

import torch

from torch_geometric.data import Data

from tqdm import tqdm

from sing import read_failed_log, parse_silph, ShareAssignmentModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-dir",
    type=str,
    default="dataset/raw",
    help="Path to dataset directory",
)
parser.add_argument(
    "--circuits-file",
    type=str,
    default="circuits.txt",
    help="Path to file with C source files to include",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=60,
    help="Timeout for circuit execution (in sec)",
)
parser.add_argument(
    "--binary",
    type=str,
    default="scripts/run_circ.sh",
    help="Path to run_circ.sh",
)
parser.add_argument(
    "--runs",
    type=int,
    default=10,
    help="Benchmark runs to average from",
)
parser.add_argument(
    "--output-csv",
    type=str,
    default="share_assignment.csv",
    help="Output CSV file path",
)
parser.add_argument(
    "--silph-cost-model",
    type=str,
    default="empirical",
    help="Silph cost model to use",
)
parser.add_argument(
    "--silph-selection-scheme",
    type=str,
    default="smart_glp",
    help="Silph selection scheme to use",
)

args = parser.parse_args()

share_assignment_checkpoint = torch.load("checkpoint.pt")

model = ShareAssignmentModel()
model.load_state_dict(share_assignment_checkpoint["state_dict"])
model.eval()

failed, duplicates = read_failed_log(osp.join(args.dataset_dir, "failed.log"))

c_file_paths = []
c_file_hashes = []

with open(args.circuits_file, "r") as f:
    for line in f:
        c_file_path = line.rstrip()
        c_file_path = osp.abspath(c_file_path)

        with open(c_file_path, "rb") as f:
            shahash = hashlib.file_digest(f, "sha256").hexdigest()

        if shahash in failed:
            continue

        while shahash in duplicates:
            if duplicates[shahash] in c_file_hashes:
                # a duplicate of this circuit has already been added
                break

            shahash = duplicates[shahash]
        else:
            if shahash in c_file_hashes:
                continue

            c_file_paths.append(c_file_path)
            c_file_hashes.append(shahash)


def run_sing(circuit_dir):
    cost = None
    costs = []

    for i in range(args.runs):
        bytecode_file = glob.glob(osp.join(circuit_dir, "*_c_main_bytecode.txt"))
        constant_input_file = glob.glob(osp.join(circuit_dir, "*_c_const.txt"))
        share_assignment_file = glob.glob(osp.join(circuit_dir, "*_c_share_map.txt"))

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

        costs.append((time_end - time_start) / (1000 * 1000))

    if cost is None:
        cost = mean(costs)

    return cost


def run(c_file_path):
    cost_sing = None
    cost = None
    costs = []

    time_start = time.perf_counter_ns()
    i = 0
    timeout = 1000 * 1000 * 1000 * args.timeout
    while i < args.runs and time.perf_counter_ns() - time_start < timeout:
        i += 1

        with tempfile.TemporaryDirectory() as result_dir:
            command = [
                args.binary,
                c_file_path,
                args.silph_cost_model,
                args.silph_selection_scheme,
                result_dir,
            ]

            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            try:
                out, err = process.communicate(timeout=args.timeout)

                if process.returncode != 0:
                    cost = "error"
                    break
                else:
                    lines = out.split("\n")
                    lines = list(filter(lambda x: x.startswith("LOG: Assignment time: "), lines))

                    if len(lines) == 1:
                        cost = lines[0]
                        cost = cost.replace("LOG: Assignment time: ", "")

                        if i == 1:
                            circuit_dir = glob.glob(osp.join(result_dir, "*"))

                            assert len(circuit_dir) == 1
                            cost_sing = run_sing(circuit_dir[0])
                    else:
                        cost = "error"
                        break
            except subprocess.TimeoutExpired:
                process.kill()

                cost = "timeout"
                break


    if cost is None:
        cost = mean(costs)

    return cost_sing, cost


results = []

for c_file_path, c_file_hash in tqdm(list(zip(c_file_paths, c_file_hashes))):
    cost_sing, cost_silph = run(c_file_path)

    results += [{
        "c_file_path": c_file_path,
        "shahash": c_file_hash,
        "mode": "SING",
        "runtime": cost_sing,
    }, {
        "c_file_path": c_file_path,
        "shahash": c_file_hash,
        "mode": "Silph",
        "runtime": cost_silph,
    }]

assert len(results) > 0

with open(args.output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())

    writer.writeheader()
    for result in results:
        writer.writerow(result)
