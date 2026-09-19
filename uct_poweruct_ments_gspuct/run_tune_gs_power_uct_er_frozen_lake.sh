#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"

ensure_dummy_binary

RESULT_FILE="$SCRIPT_DIR/tune_gs_power_uct_er_frozen_lake_result.txt"
# Deliberately the SAME case dir as run_tune_gs_power_uct_er.sh: case files are keyed by
# env/c/p/c2/c3, so the 10 frozen_lake combos already tuned there (c=0.5, p in {0.0, 2.5},
# c2=c3 in {0, 0.1, 0.5, 1, 2}) are reused as-is and only the new combos below actually run.
CASE_DIR="$SCRIPT_DIR/tune_gs_power_uct_er_cases"
N_EXPERIMENTS=300

mkdir -p "$CASE_DIR"

environments=(frozen_lake)

# Goal: beat the tuned gs_power_uct baseline on frozen_lake (c=0.5, p=0.0; final N=1000 run in
# run_result.txt: mean_reward=0.579 at 2048 sims, 0.528 at 1024, 0.479 at 512). The first pass
# (run_tune_gs_power_uct_er.sh) only varied (c2, c3) tied together at a single c=0.5, and every
# result there was within ~1 SE of that baseline (2std in the case files is 2*population std,
# NOT a confidence interval on the mean -- with N=300 the real SE is std/sqrt(300) ~ 0.03, so
# use analyze_tune_results.py's se~ column, not the raw 2std lines, to judge significance).
#
# This second pass widens the search along two axes that were not covered before:
#   1. The base exploration constant c: the ER bonus interacts with c (a bigger base UCB term
#      can drown out a small ER bonus, or a smaller one can let it dominate), so sweep c itself
#      instead of holding it fixed at the gs_power_uct-tuned value.
#   2. Untied (c2, c3): the previous grid only tied C2 = C3 (paper default, edge and child
#      channel equally weighted). Isolating each channel (c3=0 or c2=0) tests whether one of
#      the two resistance terms is actually responsible for any gain, and finer/larger tied
#      magnitudes (0.25 .. 4.0) fill in and extend the range where the first pass saw its only
#      hints of improvement (c2=c3=0.5 reached 0.583 at 2048, marginally above baseline).
#
# p is kept at 0.0 (matching the baseline's max backup): the first pass already showed p=2.5
# strictly worse for ER on frozen_lake, so that axis is dropped here to spend the budget on
# (c, c2, c3) instead.
c_values=(0.5 1.0)
p_values=(0.0 1.0 2.5)
coefficient_pairs=(
    "0.0 0.0"
    "0.1 0.1"
    "0.25 0.25"
    "0.5 0.5"
    "0.75 0.75"
    "1.0 1.0"
    "1.5 1.5"
    "2.0 2.0"
    "3.0 3.0"
    "4.0 4.0"
    "0.5 0.0"
    "1.0 0.0"
    "2.0 0.0"
    "0.0 0.5"
    "0.0 1.0"
    "0.0 2.0"
)

cases_per_env=$((${#c_values[@]} * ${#p_values[@]} * ${#coefficient_pairs[@]}))
total_cases=$((${#environments[@]} * cases_per_env))
completed_cases=0
ordered_case_files=()

# Rebuilt after every case so tune_gs_power_uct_er_frozen_lake_result.txt always reflects
# everything finished so far, even if the run is interrupted midway.
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

    for c_value in "${c_values[@]}"; do
        for p_value in "${p_values[@]}"; do
            for coefficient_pair in "${coefficient_pairs[@]}"; do
                read -r c2_value c3_value <<< "$coefficient_pair"
                completed_cases=$((completed_cases + 1))
                case_file="$CASE_DIR/${env_name}__c${c_value}__p${p_value}__c2_${c2_value}__c3_${c3_value}.txt"
                ordered_case_files+=("$case_file")

                if [[ -s "$case_file" ]]; then
                    echo "[$completed_cases/$total_cases] $env_name gs_power_uct_er c=$c_value p=$p_value c2=$c2_value c3=$c3_value (already done, skipping)"
                    continue
                fi

                echo "[$completed_cases/$total_cases] $env_name gs_power_uct_er c=$c_value p=$p_value c2=$c2_value c3=$c3_value"
                reset_result_file "$case_file"
                run_case "$case_file" "$N_EXPERIMENTS" gs_power_uct_er "${env_args[@]}" \
                    "$c_value" "$p_value" "$c2_value" "$c3_value"
                rebuild_aggregate
            done
        done
    done
done

rebuild_aggregate
echo "Saved summaries to $RESULT_FILE"
