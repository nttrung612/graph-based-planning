#!/usr/bin/env bash
# Final run of GS-E3W *without ER*: the gs_{ments,rents,tents}[_f]_er algorithms with C2 = C3 = 0.
# TAU and EPSILON are taken unchanged from run_final_config.sh (the ER final parameters); only the last two
# params (C2, C3) are overridden to 0.0. With c2 = c3 = 0 the _er algorithms are the paper's GS-E3W
# (lambda = eps*K/log(2 + N(s))), unlike plain gs_ments/rents/tents which keep the legacy log(N + 1).
#
# Output: run_result_e3w_c0.txt (a separate file; run_result.txt is not touched).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"
source "$SCRIPT_DIR/run_final_config.sh"

ensure_dummy_binary

RESULT_FILE="$SCRIPT_DIR/run_result_e3w_c0.txt"
N_EXPERIMENTS=1000

environments=(
    frozen_lake
    four_rooms
    passenger_grid
    factored_river_swim
    # sysadmin_ring  # enable for the depth variants only: gs_*_f_er has no sysadmin_ring parameters
)

algorithms=(
    gs_ments_er
    gs_rents_er
    gs_tents_er
    gs_ments_f_er
    gs_rents_f_er
    gs_tents_f_er
)

reset_result_file "$RESULT_FILE"

total_cases=$((${#environments[@]} * ${#algorithms[@]}))
completed_cases=0

for env_name in "${environments[@]}"; do
    build_env_args "$env_name"
    env_args=("${ENV_ARGS[@]}")

    for algorithm in "${algorithms[@]}"; do
        get_final_params "$env_name" "$algorithm"
        params=("${FINAL_PARAMS[@]}")
        if [[ ${#params[@]} -ne 4 ]]; then
            echo "Expected TAU EPSILON C2 C3 for $env_name $algorithm, got: ${params[*]}" >&2
            exit 1
        fi
        params[2]="0.0"
        params[3]="0.0"
        completed_cases=$((completed_cases + 1))
        echo "[$completed_cases/$total_cases] $env_name $algorithm params=${params[*]}"
        run_case "$RESULT_FILE" "$N_EXPERIMENTS" "$algorithm" "${env_args[@]}" "${params[@]}"
    done
done

echo "Saved summaries to $RESULT_FILE"
