#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUMMY_BINARY="$REPO_ROOT/c++/build/dummy"
MPI_CMD=(mpirun -np 8 "$DUMMY_BINARY")
SUMMARY_START='^===== Run Summary =====$'
SUMMARY_END='^=======================$'
ENV_ARGS=()
RUN_CASE_START_US=
RUN_CASE_SUMMARY_END_US=
RUN_CASE_ELAPSED_SECONDS=

ensure_dummy_binary() {
    if [[ ! -x "$DUMMY_BINARY" ]]; then
        echo "Missing executable: $DUMMY_BINARY" >&2
        echo "Build the project first so c++/build/dummy exists." >&2
        exit 1
    fi
}

reset_result_file() {
    local result_file="$1"
    : > "$result_file"
}

now_epoch_us() {
    if [[ -n "${EPOCHREALTIME-}" ]]; then
        local seconds="${EPOCHREALTIME%.*}"
        local fraction="${EPOCHREALTIME#*.}000000"
        printf '%s%s\n' "$seconds" "${fraction:0:6}"
    else
        local epoch_ns
        epoch_ns="$(date +%s%N 2>/dev/null || true)"
        if [[ "$epoch_ns" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "$((epoch_ns / 1000))"
        else
            printf '%s000000\n' "$(date +%s)"
        fi
    fi
}

format_epoch_us() {
    local epoch_us="$1"
    printf '%d.%06d' "$((epoch_us / 1000000))" "$((epoch_us % 1000000))"
}

elapsed_seconds() {
    local start_us="$1"
    local end_us="$2"
    local delta_us=$((end_us - start_us))

    if [[ $delta_us -lt 0 ]]; then
        delta_us=0
    fi

    printf '%d.%06d\n' "$((delta_us / 1000000))" "$((delta_us % 1000000))"
}

csv_field() {
    local value="$1"
    value="${value//\"/\"\"}"
    printf '"%s"' "$value"
}

reset_timing_file() {
    local timing_file="$1"
    printf 'case_index,total_cases,repeat,total_repeats,environment,algorithm,n_experiments,params,start_epoch_seconds,summary_end_epoch_seconds,elapsed_seconds\n' > "$timing_file"
}

append_timing_row() {
    local timing_file="$1"
    local case_index="$2"
    local total_cases="$3"
    local repeat_index="$4"
    local total_repeats="$5"
    local env_name="$6"
    local algorithm="$7"
    local n_experiments="$8"
    local params="$9"
    local start_us="${10}"
    local end_us="${11}"
    local elapsed="${12}"

    {
        printf '%s,%s,%s,%s,' "$case_index" "$total_cases" "$repeat_index" "$total_repeats"
        csv_field "$env_name"
        printf ','
        csv_field "$algorithm"
        printf ',%s,' "$n_experiments"
        csv_field "$params"
        printf ',%s,%s,%s\n' "$(format_epoch_us "$start_us")" "$(format_epoch_us "$end_us")" "$elapsed"
    } >> "$timing_file"
}

append_summary_block() {
    local full_output_file="$1"
    local result_file="$2"

    if ! grep -q "$SUMMARY_START" "$full_output_file"; then
        echo "Could not find Run Summary in command output." >&2
        return 1
    fi

    sed -n "/$SUMMARY_START/,/$SUMMARY_END/p" "$full_output_file" >> "$result_file"
    printf '\n' >> "$result_file"
}

run_case() {
    local result_file="$1"
    shift

    local tmp_output
    local summary_end_time_file
    tmp_output="$(mktemp)"
    summary_end_time_file="$(mktemp)"
    local -a cmd=("${MPI_CMD[@]}" "$@")

    echo "Running: ${cmd[*]}"

    RUN_CASE_START_US="$(now_epoch_us)"
    RUN_CASE_SUMMARY_END_US=
    RUN_CASE_ELAPSED_SECONDS=

    set +e
    "${cmd[@]}" 2>&1 | while IFS= read -r line || [[ -n "$line" ]]; do
        printf '%s\n' "$line"
        printf '%s\n' "$line" >> "$tmp_output"

        if [[ "$line" =~ $SUMMARY_START ]]; then
            seen_summary=1
        fi
        if [[ ${seen_summary:-0} -eq 1 && "$line" =~ $SUMMARY_END ]]; then
            now_epoch_us > "$summary_end_time_file"
            seen_summary=0
        fi
    done
    local -a pipe_status=("${PIPESTATUS[@]}")
    set -e
    local cmd_status="${pipe_status[0]}"
    local output_status="${pipe_status[1]:-0}"

    if [[ -s "$summary_end_time_file" ]]; then
        RUN_CASE_SUMMARY_END_US="$(< "$summary_end_time_file")"
    else
        RUN_CASE_SUMMARY_END_US="$(now_epoch_us)"
    fi
    RUN_CASE_ELAPSED_SECONDS="$(elapsed_seconds "$RUN_CASE_START_US" "$RUN_CASE_SUMMARY_END_US")"

    if [[ $cmd_status -ne 0 ]]; then
        rm -f "$tmp_output" "$summary_end_time_file"
        echo "Command failed with exit code $cmd_status" >&2
        return "$cmd_status"
    fi
    if [[ $output_status -ne 0 ]]; then
        rm -f "$tmp_output" "$summary_end_time_file"
        echo "Failed while capturing command output." >&2
        return "$output_status"
    fi

    local append_status=0
    append_summary_block "$tmp_output" "$result_file" || append_status=$?
    rm -f "$tmp_output" "$summary_end_time_file"
    if [[ $append_status -ne 0 ]]; then
        return "$append_status"
    fi
}

build_env_args() {
    local env_name="$1"
    ENV_ARGS=()
    case "$env_name" in
        passenger_grid)
            ENV_ARGS=("$env_name" "70")
            ;;
        frozen_lake|factored_river_swim|four_rooms|sysadmin_ring)
            ENV_ARGS=("$env_name")
            ;;
        *)
            echo "Unsupported environment: $env_name" >&2
            return 1
            ;;
    esac
}
