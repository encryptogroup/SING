import argparse

import pandas as pd
from scipy.stats import gmean


parser = argparse.ArgumentParser()
parser.add_argument(
    "--circuit-set",
    choices=["benchmark", "all"],
    default="benchmark",
    help="Circuits to include",
)
parser.add_argument(
    "--print-mode",
    choices=["factor", "percent"],
    default="factor",
    help="How to print the speedup; 2s before, 0.5s after is factor:4 and percent:-75",
)
args = parser.parse_args()

benchmark_circuits = pd.read_csv("circuits.csv")


def process_comm(raw):
    if raw == "timeout":
        return None
    else:
        return int(float(raw))


def process_time(raw):
    if raw in ["", "timeout", "error"]:
        return None
    elif raw.endswith("ms"):
        return float(raw[:-2])
    elif raw.endswith("s"):
        return float(raw[:-1]) * 1000
    else:
        return float(raw)


def read_share_assignment_results(csv_file):
    share_assignment_results = pd.read_csv(
        csv_file,
        converters={
            "runtime": process_time,
        },
    )

    return share_assignment_results


def read_mpc_results(csv_files):
    dfs = []
    for csv_file in csv_files:
        new_results = pd.read_csv(
            csv_file,
            converters={
                "runtime": process_time,
                "comm": process_comm,
            },
        )
        dfs.append(new_results)

    mpc_results = pd.concat(dfs)

    return mpc_results


def calculate_speedup(before, after):
    speedup = before / after
    speedup = speedup.dropna()

    return speedup


def print_gmean(speedup):
    mean_speedup = gmean(speedup)

    if args.print_mode == "percent":
        return ((1 / mean_speedup) - 1) * 100
    elif args.print_mode == "factor":
        return mean_speedup

def calculate_speedup_share_assignment(csv_file, filter_only=None):
    share_assignment_results = read_share_assignment_results(csv_file)

    if filter_only == "benchmarks":
        share_assignment_results = share_assignment_results.merge(benchmark_circuits, how="inner")

    share_assignment_results = (
        share_assignment_results.pivot(
            values="runtime", index="shahash", columns="mode"
        )
        .reset_index()
    )[["shahash", "Silph", "SING"]]

    speedup = calculate_speedup(share_assignment_results["Silph"], share_assignment_results["SING"])

    print(f"share assignment: geometric mean speedup over {len(speedup)} benchmarks: {print_gmean(speedup)}")


def calculate_speedup_mpc(csv_files, y, prefix):
    mpc_results = read_mpc_results(csv_files)

    if y not in ["runtime", "comm"]:
        raise RuntimeError("unknown mode")

    mpc_results = (
        mpc_results.pivot(values=y, index="shahash", columns="mode")
        .reset_index()
    )

    for col in ["SING 1", "SING 2", "SING 3"]:
        speedup = calculate_speedup(mpc_results["Silph"], mpc_results[col])

        print(f"{prefix}{y}: geometric mean speedup of {col} over {len(speedup)} benchmarks: {print_gmean(speedup)}")


if args.circuit_set == "benchmark":
    calculate_speedup_share_assignment("share-assignment/smart_glp.csv", filter_only="benchmarks")

    lan_files = [
        "mpc/silph_LAN.csv",
        # SING 1: supervised
        "mpc/a3a46d07-dbfb-4db4-a602-c6a0d107a6b0_LAN.csv",
        # SING 2: semi-supervised with C_Silph
        "mpc/7cd769bf-53bf-443f-963f-c289ccc17cbd_LAN.csv",
        # SING 3: semi-supervised with C_runtime_WAN
        "mpc/0ab0a50c-5ab2-4fa3-9d7a-fbfd9715007d_LAN.csv",
    ]

    wan_files = list(map(lambda x: x.replace("LAN", "WAN"), lan_files))
elif args.circuit_set == "all":
    calculate_speedup_share_assignment("share-assignment-all/smart_glp.csv.gz")

    lan_files = [
        "mpc-all/silph_LAN.csv.gz",
        # SING 1: supervised
        "mpc-all/a3a46d07-dbfb-4db4-a602-c6a0d107a6b0_LAN.csv.gz",
        # SING 2: semi-supervised with C_Silph
        "mpc-all/7cd769bf-53bf-443f-963f-c289ccc17cbd_LAN.csv.gz",
        # SING 3: semi-supervised with C_runtime_WAN
        "mpc-all/0ab0a50c-5ab2-4fa3-9d7a-fbfd9715007d_LAN.csv.gz",
    ]

    wan_files = list(map(lambda x: x.replace("LAN", "WAN"), lan_files))

print()

calculate_speedup_mpc(lan_files, "comm", "")
calculate_speedup_mpc(lan_files, "runtime", "LAN ")
calculate_speedup_mpc(wan_files, "runtime", "WAN ")
