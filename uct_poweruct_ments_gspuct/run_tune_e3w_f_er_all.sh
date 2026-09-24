#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"

ensure_dummy_binary

N_EXPERIMENTS=300
algorithms=(gs_ments_f_er gs_rents_f_er gs_tents_f_er)
environments=(frozen_lake four_rooms passenger_grid factored_river_swim sysadmin_ring)
taus=(0.01 0.25)
default_epsilons=(0.5)
# With 21 actions, epsilon=0.5 keeps lambda clipped to 1 throughout the current
# rollout budgets in sysadmin_ring. Include smaller values to exercise the bonus.
sysadmin_ring_epsilons=(0.05 0.1 0.5)
coefficient_pairs=("0.0 0.0" "0.1 0.1" "0.5 0.5" "1.0 1.0" "2.0 2.0")

total_cases=0
for env_name in "${environments[@]}"; do
    if [[ "$env_name" == "sysadmin_ring" ]]; then
        n_eps=${#sysadmin_ring_epsilons[@]}
    else
        n_eps=${#default_epsilons[@]}
    fi
    total_cases=$((total_cases + ${#algorithms[@]} * ${#taus[@]} * n_eps * ${#coefficient_pairs[@]}))
done
completed_cases=0

rebuild_aggregate() {
    : > "$RESULT_FILE"
    for case_file in "${ordered_case_files[@]}"; do
        if [[ -s "$case_file" ]]; then
            cat "$case_file" >> "$RESULT_FILE"
        fi
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
        if [[ "$env_name" == "sysadmin_ring" ]]; then
            epsilons=("${sysadmin_ring_epsilons[@]}")
        else
            epsilons=("${default_epsilons[@]}")
        fi

        for tau in "${taus[@]}"; do
            for epsilon in "${epsilons[@]}"; do
                for coefficient_pair in "${coefficient_pairs[@]}"; do
                    read -r c2_value c3_value <<< "$coefficient_pair"
                    completed_cases=$((completed_cases + 1))
                    case_file="$CASE_DIR/${env_name}__tau${tau}__eps${epsilon}__c2_${c2_value}__c3_${c3_value}.txt"
                    ordered_case_files+=("$case_file")

                    if [[ -s "$case_file" ]]; then
                        echo "[$completed_cases/$total_cases] $env_name $algorithm tau=$tau epsilon=$epsilon c2=$c2_value c3=$c3_value (already done, skipping)"
                        continue
                    fi

                    echo "[$completed_cases/$total_cases] $env_name $algorithm tau=$tau epsilon=$epsilon c2=$c2_value c3=$c3_value"
                    reset_result_file "$case_file"
                    run_case "$case_file" "$N_EXPERIMENTS" "$algorithm" "${env_args[@]}" \
                        "$tau" "$epsilon" "$c2_value" "$c3_value"
                    rebuild_aggregate
                done
            done
        done
    done

    rebuild_aggregate
    echo "Saved summaries to $RESULT_FILE"
done
