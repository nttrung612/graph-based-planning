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
epsilons=(0.5)
coefficient_pairs=("0.0 0.0" "0.1 0.1" "0.5 0.5" "1.0 1.0" "2.0 2.0")

cases_per_env=$((${#taus[@]} * ${#epsilons[@]} * ${#coefficient_pairs[@]}))
total_cases=$((${#environments[@]} * cases_per_env))
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
