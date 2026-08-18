#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"
source "$SCRIPT_DIR/run_final_config.sh"

ensure_dummy_binary

RESULT_FILE="$SCRIPT_DIR/run_result.txt"
N_EXPERIMENTS=1000

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
    ments
    gs_power_uct
    gs_power_uct_f
)

reset_result_file "$RESULT_FILE"

# total_cases=20
total_cases=$((${#environments[@]} * ${#algorithms[@]}))
completed_cases=0

for env_name in "${environments[@]}"; do
    build_env_args "$env_name"
    env_args=("${ENV_ARGS[@]}")

    for algorithm in "${algorithms[@]}"; do
        get_final_params "$env_name" "$algorithm"
        params=("${FINAL_PARAMS[@]}")
        completed_cases=$((completed_cases + 1))
        echo "[$completed_cases/$total_cases] $env_name $algorithm params=${params[*]}"
        run_case "$RESULT_FILE" "$N_EXPERIMENTS" "$algorithm" "${env_args[@]}" "${params[@]}"
    done
done

echo "Saved summaries to $RESULT_FILE"
