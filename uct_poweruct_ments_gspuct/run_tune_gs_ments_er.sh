#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"

ensure_dummy_binary

RESULT_FILE="$SCRIPT_DIR/tune_gs_ments_er_result.txt"
N_EXPERIMENTS=300

environments=(
    frozen_lake
    four_rooms
    passenger_grid
    factored_river_swim
    sysadmin_ring
)

# Same grid as the gs_ments_er/gs_rents_er/gs_tents_er block in run_tune.sh:
# tau narrowed to {0.01, 0.25}, epsilon fixed at 0.5, (C2, C3) tied over {0, 0.1, 0.5, 1, 2}.
# "0.0 0.0" is the unperturbed GS-E3W baseline row.
taus=(0.01 0.25)
epsilons=(0.5)
coefficient_pairs=("0.0 0.0" "0.1 0.1" "0.5 0.5" "1.0 1.0" "2.0 2.0")

reset_result_file "$RESULT_FILE"

cases_per_env=$((${#taus[@]} * ${#epsilons[@]} * ${#coefficient_pairs[@]}))
total_cases=$((${#environments[@]} * cases_per_env))
completed_cases=0

for env_name in "${environments[@]}"; do
    build_env_args "$env_name"
    env_args=("${ENV_ARGS[@]}")

    for tau in "${taus[@]}"; do
        for epsilon in "${epsilons[@]}"; do
            for coefficient_pair in "${coefficient_pairs[@]}"; do
                read -r c2_value c3_value <<< "$coefficient_pair"
                completed_cases=$((completed_cases + 1))
                echo "[$completed_cases/$total_cases] $env_name gs_ments_er tau=$tau epsilon=$epsilon c2=$c2_value c3=$c3_value"
                run_case "$RESULT_FILE" "$N_EXPERIMENTS" gs_ments_er "${env_args[@]}" \
                    "$tau" "$epsilon" "$c2_value" "$c3_value"
            done
        done
    done
done

echo "Saved summaries to $RESULT_FILE"
