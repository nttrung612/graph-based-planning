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

# GS-Power-UCT-ER. The (c, p) grid is deliberately narrow: the base exploration parameters are
# already covered by the gs_power_uct sweep above, and the ER coefficients are what needs tuning.
# C2 = C3 is the paper's default (edge and child channel equally weighted); to test the channels
# separately, add pairs such as ("0.5" "0.0") or ("0.0" "0.5") to er_coefficient_pairs.
er_c_values=(0.5)
er_p_values=(0.0 2.5)
er_coefficient_pairs=("0.0 0.0" "0.1 0.1" "0.5 0.5" "1.0 1.0" "2.0 2.0")

# GS-E3W-ER. The "0.0 0.0" pair above is the paper's own GS-E3W (same lambda schedule, no bonus),
# so it doubles as the apples-to-apples baseline row for the ER ablation.
e3w_er_algorithms=(gs_ments_er gs_rents_er gs_tents_er)
e3w_er_taus=(0.01 0.25)
e3w_er_epsilons=(0.5)

reset_result_file "$RESULT_FILE"

cases_per_env=$((${#max_uct_alphas[@]} +
                 3 * ${#shared_alphas[@]} * ${#shared_p_values[@]} +
                 ${#ments_taus[@]} * ${#ments_epsilons[@]} +
                 ${#er_c_values[@]} * ${#er_p_values[@]} * ${#er_coefficient_pairs[@]} +
                 ${#e3w_er_algorithms[@]} * ${#e3w_er_taus[@]} * ${#e3w_er_epsilons[@]} *
                     ${#er_coefficient_pairs[@]}))
total_cases=$((${#environments[@]} * cases_per_env))
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

    for c_value in "${er_c_values[@]}"; do
        for p_value in "${er_p_values[@]}"; do
            for coefficient_pair in "${er_coefficient_pairs[@]}"; do
                read -r c2_value c3_value <<< "$coefficient_pair"
                completed_cases=$((completed_cases + 1))
                echo "[$completed_cases/$total_cases] $env_name gs_power_uct_er c=$c_value p=$p_value c2=$c2_value c3=$c3_value"
                run_case "$RESULT_FILE" "$N_EXPERIMENTS" gs_power_uct_er "${env_args[@]}" \
                    "$c_value" "$p_value" "$c2_value" "$c3_value"
            done
        done
    done

    for algorithm in "${e3w_er_algorithms[@]}"; do
        for tau in "${e3w_er_taus[@]}"; do
            for epsilon in "${e3w_er_epsilons[@]}"; do
                for coefficient_pair in "${er_coefficient_pairs[@]}"; do
                    read -r c2_value c3_value <<< "$coefficient_pair"
                    completed_cases=$((completed_cases + 1))
                    echo "[$completed_cases/$total_cases] $env_name $algorithm tau=$tau epsilon=$epsilon c2=$c2_value c3=$c3_value"
                    run_case "$RESULT_FILE" "$N_EXPERIMENTS" "$algorithm" "${env_args[@]}" \
                        "$tau" "$epsilon" "$c2_value" "$c3_value"
                done
            done
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
