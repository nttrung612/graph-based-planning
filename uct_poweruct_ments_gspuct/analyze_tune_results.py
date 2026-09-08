#!/usr/bin/env python3
"""Parse tune_*_result.txt files and report the best hyperparameter combo
per (algorithm, environment).

Usage:
    python3 analyze_tune_results.py [file1.txt file2.txt ...]

With no arguments, parses every tune_*_result.txt in the current directory.
"""

import glob
import math
import re
import sys
from collections import defaultdict

SUMMARY_RE = re.compile(r"===== Run Summary =====\n(.*?)\n=======================", re.S)
SIM_LINE_RE = re.compile(
    r"simulations=(\d+), mean_reward=([-\d.eE]+), std_reward=([-\d.eE]+), 2std=([-\d.eE]+)"
)

# Keys that identify a case but are not "hyperparameters" per se.
NON_PARAM_KEYS = {"algorithm", "environment", "n_experiments"}


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
        params = {k: v for k, v in meta.items() if k not in NON_PARAM_KEYS}

        cases.append(
            {
                "algorithm": algorithm,
                "environment": environment,
                "n_experiments": n_experiments,
                "params": params,
                "entries": entries,
                "source": path,
            }
        )

    return cases


def params_str(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def summarize_case(case):
    entries = case["entries"]
    n = max(case["n_experiments"], 1)
    # Rank by the sum of mean_reward across every rollout budget in this case's
    # sweep, rewarding combos that perform well across the whole budget range
    # rather than just at one point.
    score = sum(e["mean_reward"] for e in entries)
    # Approximate SE of the sum by adding per-budget variances; budgets reuse the
    # same episode seeds across a case (only the rollout count differs) so this
    # slightly overstates the true uncertainty, but it is a fine tie-break signal.
    se = math.sqrt(sum((e["std_reward"] / math.sqrt(n)) ** 2 for e in entries))
    last_budget = max(entries, key=lambda e: e["simulations"])
    return score, se, last_budget


def main():
    paths = sys.argv[1:] or sorted(glob.glob("tune_*_result.txt"))
    if not paths:
        print("No tune_*_result.txt files found.", file=sys.stderr)
        sys.exit(1)

    all_cases = []
    for path in paths:
        all_cases.extend(parse_file(path))

    grouped = defaultdict(list)
    for case in all_cases:
        grouped[(case["algorithm"], case["environment"])].append(case)

    for (algorithm, environment) in sorted(grouped):
        cases = grouped[(algorithm, environment)]
        scored = []
        for case in cases:
            score, se, last_budget = summarize_case(case)
            scored.append((score, se, last_budget, case))
        scored.sort(key=lambda t: t[0], reverse=True)

        best_score, best_se, best_last, best_case = scored[0]
        n_budgets = len(best_case["entries"])
        print(f"### {algorithm} / {environment}  ({len(scored)} combos, ranked by sum of mean_reward over {n_budgets} budgets)")
        print(
            f"  BEST: {params_str(best_case['params'])}"
            f"  | total_reward={best_score:.4f} (se~{best_se:.4f})"
        )

        # Flag runner-ups whose score is within ~1 combined SE of the best
        # (i.e. likely not a statistically meaningful difference).
        for score, se, last_budget, case in scored[1:4]:
            gap = best_score - score
            combined_se = math.sqrt(best_se**2 + se**2)
            tag = "within noise of BEST" if gap < combined_se else "clearly worse"
            print(
                f"    runner-up: {params_str(case['params'])}"
                f"  | total_reward={score:.4f}  | {tag}"
            )
        print()


if __name__ == "__main__":
    main()
