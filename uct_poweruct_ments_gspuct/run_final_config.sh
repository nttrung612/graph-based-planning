#!/usr/bin/env bash

# Edit the values below after reading tune_result.txt.
# Each entry corresponds to one (environment, algorithm) pair.
#
# The gs_*_er rows come from analyze_tune_results.py over tune_gs_{power_uct,ments,rents,tents}_er_result.txt
# (BEST combo per (algorithm, environment), ranked by summed mean_reward across the rollout-budget sweep).
# Exception: whenever the top-ranked combo had c2 = c3 = 0 (i.e. the ER bonus contributes nothing, making
# the row indistinguishable from the non-ER algorithm), the runner-up combo is used instead so the final
# run actually exercises the ER term. That substitution applies to gs_ments_er:sysadmin_ring and
# gs_power_uct_er:four_rooms below.

FINAL_PARAMS=()

get_final_params() {
    local env_name="$1"
    local algorithm="$2"
    FINAL_PARAMS=()

    case "${env_name}:${algorithm}" in
       

        frozen_lake:max_uct) FINAL_PARAMS=("1.5") ;;
        frozen_lake:power_uct) FINAL_PARAMS=("1.5" "2.0") ;;
        frozen_lake:ments) FINAL_PARAMS=("0.25" "0.5") ;;
        frozen_lake:gs_power_uct) FINAL_PARAMS=("0.5" "0.0") ;;
        frozen_lake:gs_power_uct_f) FINAL_PARAMS=("1.25" "0.0") ;;
        frozen_lake:gs_power_uct_er) FINAL_PARAMS=("0.5" "0.0" "1.0" "1.0") ;;
        frozen_lake:gs_ments_er) FINAL_PARAMS=("0.01" "0.5" "0.5" "0.5") ;;
        frozen_lake:gs_rents_er) FINAL_PARAMS=("0.01" "0.5" "0.1" "0.1") ;;
        frozen_lake:gs_tents_er) FINAL_PARAMS=("0.01" "0.5" "0.1" "0.1") ;;

        four_rooms:max_uct) FINAL_PARAMS=("1.0") ;;
        four_rooms:power_uct) FINAL_PARAMS=("1.0" "3.0") ;;
        four_rooms:ments) FINAL_PARAMS=("0.01" "0.5") ;;
        four_rooms:gs_power_uct) FINAL_PARAMS=("0.5" "0.0") ;;
        four_rooms:gs_power_uct_f) FINAL_PARAMS=("0.5" "4.0") ;;
        # BEST (c=0.5, p=2.5, c2=0.0, c3=0.0) had c2=c3=0; using runner-up (c2=2.0, c3=2.0) instead.
        four_rooms:gs_power_uct_er) FINAL_PARAMS=("0.5" "2.5" "2.0" "2.0") ;;
        four_rooms:gs_ments_er) FINAL_PARAMS=("0.01" "0.5" "0.1" "0.1") ;;
        four_rooms:gs_rents_er) FINAL_PARAMS=("0.25" "0.5" "0.1" "0.1") ;;
        four_rooms:gs_tents_er) FINAL_PARAMS=("0.01" "0.5" "0.1" "0.1") ;;

        passenger_grid:max_uct) FINAL_PARAMS=("1.5") ;;
        passenger_grid:power_uct) FINAL_PARAMS=("1.5" "2.5") ;;
        passenger_grid:ments) FINAL_PARAMS=("0.01" "0.5") ;;
        passenger_grid:gs_power_uct) FINAL_PARAMS=("0.5" "0.0") ;;
        passenger_grid:gs_power_uct_f) FINAL_PARAMS=("0.5" "4.0") ;;
        passenger_grid:gs_power_uct_er) FINAL_PARAMS=("0.5" "0.0" "2.0" "2.0") ;;
        passenger_grid:gs_ments_er) FINAL_PARAMS=("0.01" "0.5" "1.0" "1.0") ;;
        passenger_grid:gs_rents_er) FINAL_PARAMS=("0.01" "0.5" "0.1" "0.1") ;;
        passenger_grid:gs_tents_er) FINAL_PARAMS=("0.25" "0.5" "0.1" "0.1") ;;

        factored_river_swim:max_uct) FINAL_PARAMS=("0.5");;
        factored_river_swim:power_uct) FINAL_PARAMS=("0.5" "2.5") ;;
        factored_river_swim:ments) FINAL_PARAMS=("0.25" "0.5") ;;
        factored_river_swim:gs_power_uct) FINAL_PARAMS=("0.5" "2.5") ;;
        factored_river_swim:gs_power_uct_f) FINAL_PARAMS=("0.5" "2.5") ;;
        factored_river_swim:gs_power_uct_er) FINAL_PARAMS=("0.5" "0.0" "0.1" "0.1") ;;
        factored_river_swim:gs_ments_er) FINAL_PARAMS=("0.01" "0.5" "2.0" "2.0") ;;
        factored_river_swim:gs_rents_er) FINAL_PARAMS=("0.25" "0.5" "2.0" "2.0") ;;
        factored_river_swim:gs_tents_er) FINAL_PARAMS=("0.25" "0.5" "1.0" "1.0") ;;

        sysadmin_ring:max_uct) FINAL_PARAMS=("1.5");;
        sysadmin_ring:power_uct) FINAL_PARAMS=("0.5" "3.0") ;;
        sysadmin_ring:ments) FINAL_PARAMS=("0.25" "0.5") ;;
        sysadmin_ring:gs_power_uct) FINAL_PARAMS=("0.5" "2.5") ;;
        sysadmin_ring:gs_power_uct_f) FINAL_PARAMS=("0.75" "2.5") ;;
        sysadmin_ring:gs_power_uct_er) FINAL_PARAMS=("0.5" "2.5" "0.1" "0.1") ;;
        # BEST (tau=0.01, epsilon=0.05, c2=0.0, c3=0.0) had c2=c3=0; using runner-up (c2=0.1, c3=0.1) instead.
        sysadmin_ring:gs_ments_er) FINAL_PARAMS=("0.01" "0.05" "0.1" "0.1") ;;
        sysadmin_ring:gs_rents_er) FINAL_PARAMS=("0.25" "0.05" "1.0" "1.0") ;;
        sysadmin_ring:gs_tents_er) FINAL_PARAMS=("0.01" "0.05" "0.5" "0.5") ;;

        *)
            echo "Missing fixed parameters for ${env_name} ${algorithm}" >&2
            return 1
            ;;
    esac
}
