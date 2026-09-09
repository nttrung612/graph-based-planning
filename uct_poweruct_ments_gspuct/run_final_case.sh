#!/usr/bin/env bash

# Re-run a single (environment, algorithm) case from run_final.sh and append its
# summary to the existing run_result.txt, without touching any other case.
#
# plot_final_results.py keeps the LAST matching (environment, algorithm) block it
# finds in the file, so appending a fresh run is enough to override a stale one —
# no need to edit/truncate run_result.txt by hand.
#
# Usage:
#   bash run_final_case.sh ENV ALGORITHM [PARAM...]
#
# With no PARAM given, the tuned FINAL_PARAMS from run_final_config.sh is used
# (edit that file first if you want to change the hyperparameters permanently).
# Pass PARAM explicitly to try a one-off value without editing the config, e.g.:
#   bash run_final_case.sh factored_river_swim gs_power_uct 0.5 3.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"
source "$SCRIPT_DIR/run_final_config.sh"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 ENV ALGORITHM [PARAM...]" >&2
    exit 1
fi

env_name="$1"
algorithm="$2"
shift 2

ensure_dummy_binary

RESULT_FILE="$SCRIPT_DIR/run_result.txt"
N_EXPERIMENTS=1000

build_env_args "$env_name"
env_args=("${ENV_ARGS[@]}")

if [[ $# -gt 0 ]]; then
    params=("$@")
else
    get_final_params "$env_name" "$algorithm"
    params=("${FINAL_PARAMS[@]}")
fi

echo "Running single case: $env_name $algorithm params=${params[*]}"
run_case "$RESULT_FILE" "$N_EXPERIMENTS" "$algorithm" "${env_args[@]}" "${params[@]}"
echo "Appended fresh summary for $env_name/$algorithm to $RESULT_FILE"
