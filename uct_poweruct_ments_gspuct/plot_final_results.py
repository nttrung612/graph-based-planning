#!/usr/bin/env python3
"""Plot run_result.txt (output of run_final.sh) in the style of Figure 1 of the paper:
one subplot per environment, reward vs. simulation budget (log x-axis), one line per
algorithm. Graph-based algorithms are drawn as solid/thicker lines, tree-based
baselines as lighter dashed/dotted lines.

Usage:
    python3 plot_final_results.py [run_result.txt] [-o final_results.png]
"""

import argparse
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMMARY_RE = re.compile(r"===== Run Summary =====\n(.*?)\n=======================", re.S)
SIM_LINE_RE = re.compile(
    r"simulations=(\d+), mean_reward=([-\d.eE]+), std_reward=([-\d.eE]+), 2std=([-\d.eE]+)"
)
NON_PARAM_KEYS = {"algorithm", "environment", "n_experiments"}

# Display order (top-to-bottom in the legend) and labels, matching the paper's naming.
ALGO_DISPLAY = {
    "max_uct": "UCT",
    "power_uct": "POWER-UCT",
    "ments": "MENTS",
    "gs_power_uct": "GS-POWER-UCT",
    "gs_power_uct_f": "GS-POWER-UCT-F",
    "gs_power_uct_er": "GS-POWER-UCT-ER",
    "gs_ments_er": "GS-MENTS-ER",
    "gs_rents_er": "GS-RENTS-ER",
    "gs_tents_er": "GS-TENTS-ER",
}

# Tree-based baselines: lighter, dashed/dotted, matching Figure 1's convention.
TREE_STYLE = {
    "max_uct": dict(color="#7f7f7f", linestyle="--", marker="v", linewidth=1.5, alpha=0.8),
    "power_uct": dict(color="#7f7f7f", linestyle="--", marker="*", linewidth=1.5, alpha=0.8),
    "ments": dict(color="#7f7f7f", linestyle=":", marker="^", linewidth=1.5, alpha=0.8),
}

# Graph-based methods: solid, thicker, saturated colors.
GRAPH_STYLE = {
    "gs_power_uct": dict(color="#1f77b4", linestyle="-", marker="^", linewidth=2.5),
    "gs_power_uct_f": dict(color="#ff7f0e", linestyle="-", marker="o", linewidth=2.5),
    "gs_power_uct_er": dict(color="#2ca02c", linestyle="-", marker="s", linewidth=2.5),
    "gs_ments_er": dict(color="#d62728", linestyle="-", marker="D", linewidth=2.5),
    "gs_rents_er": dict(color="#9467bd", linestyle="-", marker="P", linewidth=2.5),
    "gs_tents_er": dict(color="#8c564b", linestyle="-", marker="X", linewidth=2.5),
}

STYLE = {**TREE_STYLE, **GRAPH_STYLE}

ENV_DISPLAY = {
    "frozen_lake": "FrozenLake",
    "passenger_grid": "PassengerGrid",
    "factored_river_swim": "FactoredRiverSwim",
    "sysadmin_ring": "SysadminRing",
    "four_rooms": "FourRooms",
}

# Subplot layout matching Figure 1 (3 top, 2 bottom).
ENV_ORDER = ["frozen_lake", "passenger_grid", "factored_river_swim", "sysadmin_ring", "four_rooms"]


def parse_file(path):
    with open(path) as f:
        text = f.read()

    cases = []
    for block in SUMMARY_RE.findall(text):
        lines = block.strip().splitlines()
        meta = {}
        sim_start = None
        for i, line in enumerate(lines):
            if line.strip() == "mean_performance_by_simulations:":
                sim_start = i + 1
                break
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        if sim_start is None:
            continue

        entries = []
        for line in lines[sim_start:]:
            m = SIM_LINE_RE.search(line)
            if m:
                entries.append(
                    {
                        "simulations": int(m.group(1)),
                        "mean_reward": float(m.group(2)),
                        "std_reward": float(m.group(3)),
                    }
                )
        if not entries:
            continue

        algorithm = meta.get("algorithm", "?")
        environment = meta.get("environment", "?")
        n_experiments = int(meta.get("n_experiments", "0") or 0)
        cases.append(
            {
                "algorithm": algorithm,
                "environment": environment,
                "n_experiments": n_experiments,
                "entries": entries,
            }
        )
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_file", nargs="?", default="run_result.txt")
    ap.add_argument("-o", "--output", default="final_results.png")
    args = ap.parse_args()

    cases = parse_file(args.result_file)
    if not cases:
        raise SystemExit(f"No Run Summary blocks found in {args.result_file}")

    # by_env[env][algorithm] = case (last one wins if an env/algorithm pair repeats,
    # e.g. because run_final.sh was re-run and appended to the same file).
    by_env = defaultdict(dict)
    for case in cases:
        by_env[case["environment"]][case["algorithm"]] = case

    envs_present = [e for e in ENV_ORDER if e in by_env]
    envs_present += [e for e in by_env if e not in ENV_ORDER]

    n = len(envs_present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    legend_handles = {}
    for ax, env in zip(axes_flat, envs_present):
        algos_here = by_env[env]
        for algo in ALGO_DISPLAY:
            if algo not in algos_here:
                continue
            case = algos_here[algo]
            entries = sorted(case["entries"], key=lambda e: e["simulations"])
            xs = [e["simulations"] for e in entries]
            ys = [e["mean_reward"] for e in entries]
            style = STYLE.get(algo, dict(color="black", linestyle="-", marker="o", linewidth=2))
            (line,) = ax.plot(xs, ys, label=ALGO_DISPLAY[algo], markersize=6, **style)
            legend_handles[algo] = line

        ax.set_xscale("log", base=2)
        ax.set_title(ENV_DISPLAY.get(env, env))
        ax.set_xlabel("Simulations")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.axis("off")

    ordered_algos = [a for a in ALGO_DISPLAY if a in legend_handles]
    fig.legend(
        [legend_handles[a] for a in ordered_algos],
        [ALGO_DISPLAY[a] for a in ordered_algos],
        loc="lower center",
        ncol=min(len(ordered_algos), 5),
        bbox_to_anchor=(0.5, -0.02 if nrows == 2 else -0.06),
        frameon=False,
    )
    fig.suptitle("Planning performance versus simulation budget", y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
