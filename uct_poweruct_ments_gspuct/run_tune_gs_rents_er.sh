#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"

ensure_dummy_binary

RESULT_FILE="$SCRIPT_DIR/tune_gs_rents_er_result.txt"
CASE_DIR="$SCRIPT_DIR/tune_gs_rents_er_cases"
N_EXPERIMENTS=300

mkdir -p "$CASE_DIR"

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
default_epsilons=(0.5)
# sysadmin_ring has K=21 actions (SYSADMIN_RING_NUM_COMPUTERS+1). The E3W-ER explore
# schedule is lambda = epsilon*K/log(N(s)+2), clipped to max_explore_prob=1.0. At
# epsilon=0.5 that gives lambda >= 1 for every N(s) up to ~36000 (see mc_graph_e3w.h
# sample_action), i.e. for the whole tested rollout-budget range (16..2048) -- so
# sample_action always takes the pure-uniform-random branch and the c2/c3-perturbed
# policy is never consulted. That is confirmed by tune_gs_rents_er_result.txt: every
# (C2, C3) row for sysadmin_ring is bit-identical. Smaller epsilon values let lambda
# drop below 1 within the tested budget, so the ER coefficients actually get exercised.
sysadmin_ring_epsilons=(0.05 0.1 0.5)
coefficient_pairs=("0.0 0.0" "0.1 0.1" "0.5 0.5" "1.0 1.0" "2.0 2.0")

epsilons_for_env() {
    if [[ "$1" == "sysadmin_ring" ]]; then
        printf '%s\n' "${sysadmin_ring_epsilons[@]}"
    else
        printf '%s\n' "${default_epsilons[@]}"
    fi
}

total_cases=0
for env_name in "${environments[@]}"; do
    n_eps=$(epsilons_for_env "$env_name" | wc -l)
    total_cases=$((total_cases + ${#taus[@]} * n_eps * ${#coefficient_pairs[@]}))
done
completed_cases=0
ordered_case_files=()

# Rebuilt after every case so tune_gs_rents_er_result.txt always reflects everything
# finished so far, even if the run is interrupted midway.
rebuild_aggregate() {
    : > "$RESULT_FILE"
    for case_file in "${ordered_case_files[@]}"; do
        if [[ -s "$case_file" ]]; then
            cat "$case_file" >> "$RESULT_FILE"
        fi
    done
}

for env_name in "${environments[@]}"; do
    build_env_args "$env_name"
    env_args=("${ENV_ARGS[@]}")
    mapfile -t epsilons < <(epsilons_for_env "$env_name")

    for tau in "${taus[@]}"; do
        for epsilon in "${epsilons[@]}"; do
            for coefficient_pair in "${coefficient_pairs[@]}"; do
                read -r c2_value c3_value <<< "$coefficient_pair"
                completed_cases=$((completed_cases + 1))
                case_file="$CASE_DIR/${env_name}__tau${tau}__eps${epsilon}__c2_${c2_value}__c3_${c3_value}.txt"
                ordered_case_files+=("$case_file")

                # A case file only ever gets content once its mpirun run has finished and the
                # summary block was parsed (see run_case/append_summary_block in
                # run_batch_common.sh); a truncated-but-empty file means a prior attempt was
                # interrupted before finishing, so -s correctly forces a re-run of that case.
                if [[ -s "$case_file" ]]; then
                    echo "[$completed_cases/$total_cases] $env_name gs_rents_er tau=$tau epsilon=$epsilon c2=$c2_value c3=$c3_value (already done, skipping)"
                    continue
                fi

                echo "[$completed_cases/$total_cases] $env_name gs_rents_er tau=$tau epsilon=$epsilon c2=$c2_value c3=$c3_value"
                reset_result_file "$case_file"
                run_case "$case_file" "$N_EXPERIMENTS" gs_rents_er "${env_args[@]}" \
                    "$tau" "$epsilon" "$c2_value" "$c3_value"
                rebuild_aggregate
            done
        done
    done
done

rebuild_aggregate
echo "Saved summaries to $RESULT_FILE"
