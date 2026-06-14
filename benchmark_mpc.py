import os
import os.path as osp

import argparse
import subprocess
import glob
import json
import tempfile
import time
import csv
from statistics import mean
import concurrent.futures
import threading
import sqlite3
import shlex

import torch
import torch.nn.functional as F

from torch_geometric.data import Data

from tqdm import tqdm

from sing import (
    read_hashes,
    read_failed_log,
    parse_silph,
    ShareAssignmentModel,
    CostModel,
    read_share_assignment_wires,
    save_share_assignment_text,
    postprocess_share_assignment,
    load_metadata,
    check_assignment_filter,
    assignment_filters,
)

import network
from ntfy import notify

splits = ["train", "val", "test"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoint.pt",
    help="Benchmark mode: Path to model checkpoint",
)
parser.add_argument(
    "--dataset-dir",
    type=str,
    default="dataset/raw",
    help="Path to dataset directory",
)
parser.add_argument(
    "--hashes-file",
    type=str,
    help="Path to text file with hashes",
)
parser.add_argument(
    "--mode",
    choices=["dataset", "benchmark"],
    default="dataset",
    help="Perform dataset generation or model benchmarking",
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
    default="ABY-vendor/build/bin/aby_interpreter",
    help="Path to aby_interpreter",
)
parser.add_argument(
    "--rerun",
    choices=["none", "timeout", "all"],
    default="none",
    help="Rerun specificbenchmarks",
)
parser.add_argument(
    "--metric",
    choices=["runtime", "communication"],
    default="runtime",
    help="Which metric to measure",
)
parser.add_argument(
    "--cost-name",
    type=str,
    default="cost",
    help="Postfix to add to the text file containing the cost for each circuit",
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
    help="Benchmark mode: Output CSV file path. Defaults to {RUN_UUID}_{NETWORK_SETTING}.csv",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Don't run benchmarks, just print what would have been run",
)
parser.add_argument(
    "--parallel",
    action="store_true",
    help="Run benchmarks in parallel",
)
parser.add_argument(
    "--max-jobs",
    type=int,
    default=8,
    help="Number of maximum parallel jobs",
)
parser.add_argument(
    "--network-setting",
    type=str,
    default="loopback",
    help="Network setting to run benchmarks in",
)
parser.add_argument(
    "--assignment-filter",
    default="no_filter",
    help="Assignment filter to exclude specific assignments from the dataset",
)
parser.add_argument(
    "--benchmark-extra-assignments",
    action="store_true",
    help="Benchmark mode: also benchmark all_b, all_y assignments",
)
parser.add_argument(
    "--benchmark-silph",
    action="store_true",
    help="Benchmark mode: also benchmark silph assignments",
)
parser.add_argument(
    "--benchmark-label",
    type=str,
    default="SING",
    help="Benchmark mode: label for these results (i.e. mode column in output CSV)",
)

args = parser.parse_args()


def run(circuit_dir, share_assignment_file, thread_id):
    user = os.getlogin()
    command = [
        "sudo",
        "-u",
        user,
        args.binary,
        "-m",
        "mpc",
        "-f",
        circuit_dir,
        "-s",
        share_assignment_file,
    ]

    if args.dry_run:
        print(shlex.join(command))
        return None, None

    prefix = [
        "sudo",
        "ip",
        "netns",
        "exec",
    ]

    server_address = f"{network.get_ip(thread_id, 0)}"

    server_command = prefix + [f"neon{thread_id}_ns0"] + command + ["-r", "0", "-a", server_address]
    client_command = prefix + [f"neon{thread_id}_ns1"] + command + ["-r", "1", "-a", server_address]

    invalid = False
    timeout_reached = False
    runtimes = []
    comms = []

    time_start = time.perf_counter_ns()
    i = 0
    timeout = 1000 * 1000 * 1000 * args.timeout
    while i < args.runs and time.perf_counter_ns() - time_start < timeout:
        i += 1

        server_process = subprocess.Popen(
            server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        client_process = subprocess.Popen(
            client_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        try:
            server_out, server_err = server_process.communicate(timeout=args.timeout)
            client_out, client_err = client_process.communicate(timeout=10)

            if server_process.returncode != 0 or client_process.returncode != 0:
                invalid = True
                if "not implemented in arithmetic sharing" not in server_err:
                    print("Unexpected error")
                    print("circuit dir:", circuit_dir)
                    print("assignment file:", share_assignment_file)
                    print(server_out, server_err, client_out, client_err)
                break
            else:
                lines = client_out.split("\n")

                if not lines[0].startswith("LOG: Client exec time: "):
                    invalid = True
                    break

                comm_lines = list(filter(lambda x: x.startswith("Comm: "), lines))
                if not len(comm_lines) == 1:
                    invalid = True
                    break

                runtimes.append(float(lines[0].replace("LOG: Client exec time: ", "")))
                comms.append(float(comm_lines[0].replace("Comm: ", "")))
        except subprocess.TimeoutExpired:
            server_process.kill()
            client_process.kill()

            timeout_reached = True
            break

    if invalid:
        return "invalid", "invalid"
    elif timeout_reached:
        return "timeout", "timeout"
    else:
        return mean(runtimes), mean(comms)


jobs = []

thread_id_map = {}
thread_id_map_lock = threading.Lock()

temp_dir = tempfile.TemporaryDirectory()

if args.mode == "benchmark":
    share_assignment_checkpoint = torch.load(args.checkpoint)

    model = ShareAssignmentModel()
    model.load_state_dict(share_assignment_checkpoint["state_dict"])
    model.eval()

    cost_model = CostModel("empirical_costs.json")

    results = []
    results_lock = threading.Lock()

    output_csv = args.output_csv
    if output_csv is None:
        run_uuid = share_assignment_checkpoint["uuid"]
        output_csv = f"{run_uuid}_{args.network_setting}.csv"

if args.hashes_file:
    hashes = read_hashes(args.hashes_file)
else:
    dataset_meta_path = osp.join(osp.dirname(__file__), "sing", "dataset.json")
    with open(dataset_meta_path, "r") as f:
        dataset_meta = json.load(f)
    hashes = sum([dataset_meta[split] for split in splits], [])

failed, duplicates = read_failed_log(osp.join(args.dataset_dir, "failed.log"))

new_hashes = []
for shahash in hashes:
    if shahash in failed:
        continue

    while shahash in duplicates:
        shahash = duplicates[shahash]

    if shahash in new_hashes:
        continue

    new_hashes.append(shahash)

hashes = new_hashes


db_con = sqlite3.connect(osp.join(args.dataset_dir, "metadata.sqlite"))

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

    circuit_dir = osp.abspath(osp.join(args.dataset_dir, shahash))

    if args.mode == "dataset":
        assignments_dir = osp.join(args.dataset_dir, shahash, "assignments")

        generated_hashes = os.listdir(assignments_dir)
        generated_hashes = list(filter(lambda x: len(x) == 64, generated_hashes))

        for name in tqdm(generated_hashes, leave=False):
            metadata = load_metadata(shahash, name, db_con)
            if not check_assignment_filter(metadata, assignment_filters[args.assignment_filter]):
                continue

            cost_file = osp.join(assignments_dir, f"{name}_{args.cost_name}.txt")
            if osp.isfile(cost_file):
                with open(cost_file, "r") as f:
                    cost = f.readline().rstrip()

                if args.rerun == "none":
                    rerun = False
                elif args.rerun == "timeout":
                    rerun = cost == "timeout"
                elif args.rerun == "all":
                    rerun = True

                if not rerun:
                    continue

            share_assignment_file = osp.abspath(
                osp.join(assignments_dir, f"{name}.txt")
            )

            jobs.append((circuit_dir, share_assignment_file, cost_file))
    elif args.mode == "benchmark":
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

        with torch.no_grad():
            out = model(
                x=data.x,
                edge_index=data.edge_index,
            )

        y = postprocess_share_assignment(data.x, out, cost_model)

        share_assignment_wires = read_share_assignment_wires(share_assignment_file[0])

        if args.benchmark_extra_assignments:
            temp_file_b = tempfile.NamedTemporaryFile(mode="w", delete=False, dir=temp_dir.name)
            save_share_assignment_text(torch.ones_like(y), share_assignment_wires, circuit, temp_file_b.name)
            jobs.append((circuit_dir, temp_file_b.name, shahash, "b"))

            temp_file_y = tempfile.NamedTemporaryFile(mode="w", delete=False, dir=temp_dir.name)
            save_share_assignment_text(torch.ones_like(y) * 2, share_assignment_wires, circuit, temp_file_y.name)
            jobs.append((circuit_dir, temp_file_y.name, shahash, "y"))

        if args.benchmark_silph:
            jobs.append((circuit_dir, share_assignment_file[0], shahash, "Silph"))

        temp_file_sing = tempfile.NamedTemporaryFile(mode="w", delete=False, dir=temp_dir.name)
        save_share_assignment_text(y, share_assignment_wires, circuit, temp_file_sing.name)
        jobs.append((circuit_dir, temp_file_sing.name, shahash, args.benchmark_label))

db_con.close()

def benchmark_job(job):
    circuit_dir, share_assignment_file, *_ = job

    # get a thread_id in range(0, max_jobs)
    with thread_id_map_lock:
        ident = threading.get_ident()
        if ident not in thread_id_map:
            thread_id_map[ident] = len(thread_id_map)

        thread_id = thread_id_map[ident]

    runtime, comm = run(circuit_dir, share_assignment_file, thread_id)

    if not args.dry_run:
        if args.mode == "dataset":
            _, _, cost_file = job

            if args.metric == "runtime":
                cost = runtime
            else:
                cost = comm

            with open(cost_file, "w") as f:
                f.write(f"{cost}\n")
        elif args.mode == "benchmark":
            _, _, shahash, mode = job

            results.append({
                "shahash": shahash,
                "mode": mode,
                "runtime": runtime,
                "comm": comm,
            })
try:
    if not args.dry_run:
        if args.parallel:
            for thread_id in range(args.max_jobs):
                network.start_virtual_network_setting(thread_id, 2, args.network_setting)
        else:
            network.start_virtual_network_setting(0, 2, args.network_setting)

    if args.parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
            futures = [executor.submit(benchmark_job, job) for job in jobs]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(jobs)):
                pass
    else:
        for job in tqdm(jobs):
            benchmark_job(job)

    if args.mode == "benchmark" and not args.dry_run:
        assert len(results) > 0

        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())

            writer.writeheader()
            for result in results:
                writer.writerow(result)
finally:
    if not args.dry_run:
        if args.parallel:
            for thread_id in range(args.max_jobs):
                network.stop_virtual_network(thread_id, 2)
        else:
            network.stop_virtual_network(0, 2)


if not args.dry_run:
    notify("benchmark_mpc.py finished")
