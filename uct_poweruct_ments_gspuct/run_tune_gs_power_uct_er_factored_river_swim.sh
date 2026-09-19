#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_batch_common.sh"

ensure_dummy_binary

RESULT_FILE="$SCRIPT_DIR/tune_gs_power_uct_er_factored_river_swim_result.txt"
# Deliberately the SAME case dir as run_tune_gs_power_uct_er.sh: case files are keyed by
# env/c/p/c2/c3, so the 10 factored_river_swim combos already tuned there (c=0.5, p in
# {0.0, 2.5}, c2=c3 in {0, 0.1, 0.5, 1, 2}) are reused as-is and only the new combos below
# actually run.
CASE_DIR="$SCRIPT_DIR/tune_gs_power_uct_er_cases"
N_EXPERIMENTS=300

mkdir -p "$CASE_DIR"

environments=(factored_river_swim)

# From tune_gs_power_uct_er_result.txt, the existing 10-combo grid already shows p is the
# dominant factor here, not (c2, c3): at c=0.5, p=0.0 every (c2, c3) scores ~31.9-32.2 total
# reward (including c2=c3=0.0, which is bit-identical to plain gs_power_uct at p=0), while
# every p=2.5 row scores ~14.85-14.95 -- over 70 combined-SE below the p=0.0 rows. So the ER
# bonus itself is not doing much yet in that grid; p=0.0 alone already explains almost all of
# the gap over the *tuned* gs_power_uct baseline (run_final_config.sh picked p=2.5 for plain
# gs_power_uct on this env, which run_result.txt shows is actually far worse than p=0.0 --
# worth revisiting separately in run_final_config.sh).
#
# This pass keeps p=0.0 fixed (matching the best-known regime) and instead widens the two
# axes that might let the ER bonus contribute on top of that: the base exploration constant c,
# and untying (c2, c3) instead of only sweeping the tied diagonal.
c_values=(0.5 1.0)
p_values=(0.0)
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

# Rebuilt after every case so tune_gs_power_uct_er_factored_river_swim_result.txt always
# reflects everything finished so far, even if the run is interrupted midway.
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
