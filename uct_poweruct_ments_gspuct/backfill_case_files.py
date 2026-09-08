#!/usr/bin/env python3
"""Backfill tune_<algo>_cases/*.txt from an existing (already completed)
tune_<algo>_result.txt aggregate file, so the resumable run_tune_<algo>.sh
scripts recognize that work as done instead of re-running it.

This is needed when a case's aggregate result file was produced by an older,
non-resumable version of the script (single shared result file, no per-case
files) -- the resumable script's skip check only looks at the per-case files,
so without this backfill it has nothing to skip against.

Usage:
    python3 backfill_case_files.py tune_gs_ments_er_result.txt
    python3 backfill_case_files.py                # does all 4 known ER result files present
"""

import re
import sys
from pathlib import Path

BLOCK_RE = re.compile(r"(===== Run Summary =====\n.*?\n=======================)\n", re.S)
META_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*): (.*)$", re.M)

# (result filename, case dir name, filename builder) -- filename builder must exactly match
# the naming scheme in the corresponding run_tune_*.sh script.
KNOWN_FILES = {
    "tune_gs_ments_er_result.txt": "tune_gs_ments_er_cases",
    "tune_gs_rents_er_result.txt": "tune_gs_rents_er_cases",
    "tune_gs_tents_er_result.txt": "tune_gs_tents_er_cases",
    "tune_gs_power_uct_er_result.txt": "tune_gs_power_uct_er_cases",
}


def case_filename(meta):
    env = meta["environment"]
    if "tau" in meta:
        return f"{env}__tau{meta['tau']}__eps{meta['epsilon']}__c2_{meta['c2']}__c3_{meta['c3']}.txt"
    return f"{env}__c{meta['c']}__p{meta['p']}__c2_{meta['c2']}__c3_{meta['c3']}.txt"


def backfill(result_path: Path, case_dir: Path):
    text = result_path.read_text()
    case_dir.mkdir(exist_ok=True)

    written = 0
    skipped = 0
    for block in BLOCK_RE.findall(text):
        meta = dict(META_LINE_RE.findall(block))
        if "environment" not in meta:
            continue

        fname = case_filename(meta)
        target = case_dir / fname
        content = block + "\n"

        if target.exists() and target.read_text() == content:
            skipped += 1
            continue

        target.write_text(content)
        written += 1

    print(f"{result_path.name}: {written} case files written, {skipped} already up to date "
          f"-> {case_dir}/")


def main():
    args = sys.argv[1:]
    targets = args or [p for p in KNOWN_FILES if Path(p).exists()]

    if not targets:
        print("No tune_*_result.txt files found.", file=sys.stderr)
        sys.exit(1)

    for arg in targets:
        result_path = Path(arg)
        if not result_path.exists():
            print(f"Skipping {arg}: not found", file=sys.stderr)
            continue
        case_dir_name = KNOWN_FILES.get(result_path.name, result_path.stem.replace("_result", "") + "_cases")
        backfill(result_path, result_path.parent / case_dir_name)


if __name__ == "__main__":
    main()
