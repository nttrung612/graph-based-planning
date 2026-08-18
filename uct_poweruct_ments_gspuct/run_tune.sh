#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"

ensure_dummy_binary

RESULT_FILE="$SCRIPT_DIR/tune_result.txt"
N_EXPERIMENTS=300

environments=(
    
    frozen_lake
    four_rooms
    passenger_grid
    factored_river_swim
    sysadmin_ring
)

max_uct_alphas=(0.5 0.75 1.0 1.25 1.5)
shared_alphas=(0.5 0.75 1.0 1.25 1.5)
shared_p_values=(0.0 1.0 2.0 2.5 3.0 4.0)
ments_taus=(0.01 0.25 0.5 0.75 0.99)
ments_epsilons=(0.5)

reset_result_file "$RESULT_FILE"

total_cases=55
completed_cases=0

for env_name in "${environments[@]}"; do
    build_env_args "$env_name"
    env_args=("${ENV_ARGS[@]}")

    for alpha in "${max_uct_alphas[@]}"; do
        completed_cases=$((completed_cases + 1))
        echo "[$completed_cases/$total_cases] $env_name max_uct alpha=$alpha"
        run_case "$RESULT_FILE" "$N_EXPERIMENTS" max_uct "${env_args[@]}" "$alpha"
    done

    for alpha in "${shared_alphas[@]}"; do
        for p_value in "${shared_p_values[@]}"; do
            completed_cases=$((completed_cases + 1))
            echo "[$completed_cases/$total_cases] $env_name power_uct alpha=$alpha p=$p_value"
            run_case "$RESULT_FILE" "$N_EXPERIMENTS" power_uct "${env_args[@]}" "$alpha" "$p_value"
        done
        for p_value in "${shared_p_values[@]}"; do
            completed_cases=$((completed_cases + 1))
            echo "[$completed_cases/$total_cases] $env_name gs_power_uct c=$alpha p=$p_value"
            run_case "$RESULT_FILE" "$N_EXPERIMENTS" gs_power_uct "${env_args[@]}" "$alpha" "$p_value"
        done
        for p_value in "${shared_p_values[@]}"; do
            completed_cases=$((completed_cases + 1))
            echo "[$completed_cases/$total_cases] $env_name gs_power_uct_f c=$alpha p=$p_value"
            run_case "$RESULT_FILE" "$N_EXPERIMENTS" gs_power_uct_f "${env_args[@]}" "$alpha" "$p_value"
        done
    done

    for tau in "${ments_taus[@]}"; do
        for epsilon in "${ments_epsilons[@]}"; do
            completed_cases=$((completed_cases + 1))
            echo "[$completed_cases/$total_cases] $env_name ments tau=$tau epsilon=$epsilon"
            run_case "$RESULT_FILE" "$N_EXPERIMENTS" ments "${env_args[@]}" "$tau" "$epsilon"
        done
    done
done

echo "Saved summaries to $RESULT_FILE"
