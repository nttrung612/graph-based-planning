#!/usr/bin/env bash

# Driver for the full GS-E3W-ER tuning sweep: runs gs_ments_er, gs_rents_er and gs_tents_er
# back to back. Each sub-script is independently resumable (skips any case whose per-case
# result file in its tune_gs_*_er_cases/ directory is already non-empty), so if this driver
# (or the machine) dies mid-run, simply re-running it picks up exactly where it left off:
# already-finished sub-scripts finish instantly (every case skipped) and the interrupted one
# resumes from its first unfinished case.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
    run_tune_gs_ments_er.sh
    run_tune_gs_rents_er.sh
    run_tune_gs_tents_er.sh
)

for script in "${scripts[@]}"; do
    echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] Starting $script ====="
    bash "$SCRIPT_DIR/$script"
    echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] Finished $script ====="
done

echo "All GS-E3W-ER tuning sweeps finished."
