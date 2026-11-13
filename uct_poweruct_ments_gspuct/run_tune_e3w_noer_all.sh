#!/usr/bin/env bash
# Tune the plain graph-regularized variants (no ER args): gs_{ments,rents,tents} and their _f versions.
# Resumable: finished cases are skipped. Output: tune_<algo>_result.txt + tune_<algo>_cases/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"

ensure_dummy_binary

N_EXPERIMENTS=300
algorithms=(gs_ments gs_rents gs_tents gs_ments_f gs_rents_f gs_tents_f)
environments=(frozen_lake four_rooms passenger_grid factored_river_swim)
taus=(0.01 0.25)
default_epsilons=(0.5)
sysadmin_ring_epsilons=(0.05 0.1 0.5)

total_cases=0
for env_name in "${environments[@]}"; do
    if [[ "$env_name" == "sysadmin_ring" ]]; then n_eps=${#sysadmin_ring_epsilons[@]}; else n_eps=${#default_epsilons[@]}; fi
    total_cases=$((total_cases + ${#algorithms[@]} * ${#taus[@]} * n_eps))
done
completed_cases=0

rebuild_aggregate() {
    : > "$RESULT_FILE"
    for case_file in "${ordered_case_files[@]}"; do
        if [[ -s "$case_file" ]]; then cat "$case_file" >> "$RESULT_FILE"; fi
    done
}

for algorithm in "${algorithms[@]}"; do
    RESULT_FILE="$SCRIPT_DIR/tune_${algorithm}_result.txt"
    CASE_DIR="$SCRIPT_DIR/tune_${algorithm}_cases"
    ordered_case_files=()
    mkdir -p "$CASE_DIR"

    for env_name in "${environments[@]}"; do
        build_env_args "$env_name"
        env_args=("${ENV_ARGS[@]}")
        if [[ "$env_name" == "sysadmin_ring" ]]; then epsilons=("${sysadmin_ring_epsilons[@]}"); else epsilons=("${default_epsilons[@]}"); fi

        for tau in "${taus[@]}"; do
            for epsilon in "${epsilons[@]}"; do
                completed_cases=$((completed_cases + 1))
                case_file="$CASE_DIR/${env_name}__tau${tau}__eps${epsilon}.txt"
                ordered_case_files+=("$case_file")

                if [[ -s "$case_file" ]]; then
                    echo "[$completed_cases/$total_cases] $env_name $algorithm tau=$tau epsilon=$epsilon (already done, skipping)"
                    continue
                fi

                echo "[$completed_cases/$total_cases] $env_name $algorithm tau=$tau epsilon=$epsilon"
                reset_result_file "$case_file"
                run_case "$case_file" "$N_EXPERIMENTS" "$algorithm" "${env_args[@]}" "$tau" "$epsilon"
                rebuild_aggregate
            done
        done
    done

    rebuild_aggregate
    echo "Saved summaries to $RESULT_FILE"
done
