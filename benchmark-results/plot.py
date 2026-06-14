import argparse
import csv

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


parser = argparse.ArgumentParser()
parser.add_argument(
    "--format",
    choices=["html", "pdf", "latex", "pivot"],
    default="html",
    help="What kind of format to output to",
)
args = parser.parse_args()

circuits = pd.read_csv("circuits.csv")


def process_comm(raw):
    if raw == "timeout":
        return None
    else:
        return int(float(raw))


def process_time(raw):
    if raw == "timeout":
        return None
    if raw.endswith("ms"):
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
    share_assignment_results = share_assignment_results.merge(circuits, how="inner")

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
        dfs.append(new_results.merge(circuits, how="inner"))

    mpc_results = pd.concat(dfs)

    return mpc_results


def plot_share_assignment(csv_file):
    share_assignment_results = read_share_assignment_results(csv_file).sort_values(
        "runtime", ascending=False
    )

    if args.format == "latex":
        share_assignment_results = (
            share_assignment_results.pivot(
                values="runtime", index="name", columns="mode"
            )
            .reset_index()
            .sort_values("Silph", ascending=False)
        )[["name", "Silph", "SING"]]

        share_assignment_results.to_latex("share_assignment.tex", index=False)

        return

    fig = px.bar(
        share_assignment_results,
        x="name",
        y="runtime",
        color="mode",
        labels={
            "name": "Circuit",
            "runtime": "Runtime [ms]",
            "mode": "",
        },
        barmode="group",
        log_y=True,
    )

    if args.format == "html":
        fig.write_html("share_assignment.html")
    elif args.format == "pdf":
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            margin=dict(l=20, r=20, t=20, b=20),
        )

        fig.write_image("share_assignment.pdf", width=400, height=300)


def plot_mpc(csv_files, y, output_file_name):
    mpc_results = read_mpc_results(csv_files).sort_values(
        ["name", "mode"], key=lambda col: col.str.lower()
    )

    if y not in ["runtime", "comm"]:
        raise RuntimeError("unknown mode")

    if args.format == "latex":
        mpc_results["comm"] = mpc_results["comm"] / (1000 * 1000)

        mpc_results = (
            mpc_results.pivot(values=y, index="name", columns="mode")
            .sort_values("name", key=lambda col: col.str.lower())
            .rename(
                mapper={
                    "CryptoNets": "CryptoNets~\\cite{cryptonets}",
                    "MiniONN": "MiniONN~\\cite{minionn}",
                    "k-means": "\\(k\\)-means",
                },
            )
            .reset_index()
        )

        for col in ["SING 1", "SING 2", "SING 3"]:
            mpc_results[f"{col} diff"] = (
                (mpc_results[col] - mpc_results["Silph"]) / mpc_results["Silph"] * 100
            )

        mpc_results = mpc_results[
            [
                "name",
                "Silph",
                "SING 1",
                "SING 1 diff",
                "SING 2",
                "SING 2 diff",
                "SING 3",
                "SING 3 diff",
            ]
        ]

        mpc_results.to_latex(
            f"{output_file_name}.tex", index=False, na_rep="{-\\tnote{\\textdagger}}"
        )

        return
    elif args.format == "pivot":
        mpc_results = (
            mpc_results.pivot(values=y, index="name", columns="mode")
            .sort_values("name", key=lambda col: col.str.lower())
            .reset_index()
        ).dropna()

        mpc_results.to_csv(f"{output_file_name}.csv", index=False)

        return

    mpc_results = mpc_results.dropna()

    fig = px.bar(
        mpc_results,
        x="name",
        y=y,
        color="mode",
        labels={
            "name": "Circuit",
            "runtime": f"MPC {output_file_name} Runtime [s]",
            "comm": "Communication [bytes]",
            "mode": "",
        },
        barmode="group",
        log_y=True,
    )

    if args.format == "html":
        fig.write_html(f"{output_file_name}.html")
    elif args.format == "pdf":
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            margin=dict(l=20, r=20, t=20, b=20),
        )

        fig.write_image(f"{output_file_name}.pdf", width=400, height=300)


def plot_combined(share_assignment_csv_file, mpc_csv_files):
    share_assignment_results = read_share_assignment_results(share_assignment_csv_file)
    mpc_results = read_mpc_results(mpc_csv_files)

    combined = pd.merge(
        share_assignment_results,
        mpc_results.replace(r"^SING.*$", "SING", regex=True),
        how="inner",
        on=["name", "shahash", "mode"],
        suffixes=["_share_assignment", "_mpc"],
    ).dropna()

    symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "pentagon",
        "star",
    ]

    fig = go.Figure()

    for mode in ["Silph", "SING"]:
        df = combined[combined["mode"] == mode].sort_values("name")

        fig.add_trace(
            go.Scatter(
                x=df["runtime_mpc"],
                y=df["runtime_share_assignment"],
                mode="markers",
                name=mode,
                marker=dict(symbol=symbols),
            )
        )

    # add symbols to legend
    for i, circuit in enumerate(combined.sort_values("name")["name"].unique()):
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                name=circuit,
                marker=dict(
                    color="black",
                    symbol=symbols[i],
                ),
                legend="legend2",
            )
        )

    fig.update_xaxes(type="log", title=dict(text="MPC LAN Runtime [s]"))
    fig.update_yaxes(type="log", title=dict(text="Share Assignment Runtime [ms]"))

    fig.update_layout(
        legend=dict(
            title=dict(text="Share Assignment"),
        ),
        legend2=dict(
            title=dict(text="Circuit"),
        ),
    )

    if args.format == "html":
        fig.update_layout(
            legend=dict(
                y=0.9,
            ),
            legend2=dict(
                y=0.6,
            ),
        )

        fig.write_html(f"combined.html")
    elif args.format == "pdf":
        fig.update_layout(
            legend=dict(
                y=1,
            ),
            legend2=dict(
                y=0,
            ),
            margin=dict(l=20, r=20, t=20, b=20),
        )

        fig.write_image(f"combined.pdf", width=500, height=400)


plot_share_assignment("share-assignment/smart_glp.csv")

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

plot_mpc(lan_files, "comm", "Communication")
plot_mpc(lan_files, "runtime", "LAN")
plot_mpc(wan_files, "runtime", "WAN")


plot_combined(
    "share-assignment/smart_glp.csv",
    [
        "mpc/silph_LAN.csv",
        # SING 3: semi-supervised with C_runtime_WAN
        "mpc/0ab0a50c-5ab2-4fa3-9d7a-fbfd9715007d_LAN.csv",
    ],
)
