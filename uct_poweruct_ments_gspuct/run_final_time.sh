#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"
source "$SCRIPT_DIR/run_final_config.sh"

ensure_dummy_binary

RESULT_FILE="$SCRIPT_DIR/run_subresult.txt"
TIMING_FILE="$SCRIPT_DIR/run_timing.csv"
N_EXPERIMENTS=100
REPEAT=10

environments=(
    
    frozen_lake
    four_rooms
    passenger_grid
     factored_river_swim
     sysadmin_ring
)

algorithms=(
     max_uct
     power_uct
     max_entropy
    gs_power_uct
    gs_power_uct_f
)

reset_result_file "$RESULT_FILE"
reset_timing_file "$TIMING_FILE"

total_cases=$((${#environments[@]} * ${#algorithms[@]}))
total_runs=$((total_cases * REPEAT))
completed_cases=0
completed_runs=0

for env_name in "${environments[@]}"; do
    build_env_args "$env_name"
    env_args=("${ENV_ARGS[@]}")

    for algorithm in "${algorithms[@]}"; do
        get_final_params "$env_name" "$algorithm"
        params=("${FINAL_PARAMS[@]}")
        completed_cases=$((completed_cases + 1))

        for ((repeat_idx = 1; repeat_idx <= REPEAT; repeat_idx++)); do
            completed_runs=$((completed_runs + 1))
            echo "[$completed_runs/$total_runs] case [$completed_cases/$total_cases] repeat [$repeat_idx/$REPEAT] $env_name $algorithm params=${params[*]}"
            run_case "$RESULT_FILE" "$N_EXPERIMENTS" "$algorithm" "${env_args[@]}" "${params[@]}"
            append_timing_row "$TIMING_FILE" "$completed_cases" "$total_cases" "$repeat_idx" "$REPEAT" "$env_name" "$algorithm" "$N_EXPERIMENTS" "${params[*]}" "$RUN_CASE_START_US" "$RUN_CASE_SUMMARY_END_US" "$RUN_CASE_ELAPSED_SECONDS"
        done
    done
done

echo "Saved summaries to $RESULT_FILE"
echo "Saved timings to $TIMING_FILE"
