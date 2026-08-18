#!/usr/bin/env bash

# Edit the values below after reading tune_result.txt.
# Each entry corresponds to one (environment, algorithm) pair.

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

        four_rooms:max_uct) FINAL_PARAMS=("1.0") ;;
        four_rooms:power_uct) FINAL_PARAMS=("1.0" "3.0") ;;
        four_rooms:ments) FINAL_PARAMS=("0.01" "0.5") ;;
        four_rooms:gs_power_uct) FINAL_PARAMS=("0.5" "0.0") ;;
        four_rooms:gs_power_uct_f) FINAL_PARAMS=("0.5" "4.0") ;;

        passenger_grid:max_uct) FINAL_PARAMS=("1.5") ;;
        passenger_grid:power_uct) FINAL_PARAMS=("1.5" "2.5") ;;
        passenger_grid:ments) FINAL_PARAMS=("0.01" "0.5") ;;
        passenger_grid:gs_power_uct) FINAL_PARAMS=("0.5" "0.0") ;;
        passenger_grid:gs_power_uct_f) FINAL_PARAMS=("0.5" "4.0") ;;

        factored_river_swim:max_uct) FINAL_PARAMS=("0.5");;
        factored_river_swim:power_uct) FINAL_PARAMS=("0.5" "2.5") ;;
        factored_river_swim:ments) FINAL_PARAMS=("0.25" "0.5") ;;
        factored_river_swim:gs_power_uct) FINAL_PARAMS=("0.5" "2.5") ;;
        factored_river_swim:gs_power_uct_f) FINAL_PARAMS=("0.5" "2.5") ;;

        sysadmin_ring:max_uct) FINAL_PARAMS=("1.5");;
        sysadmin_ring:power_uct) FINAL_PARAMS=("0.5" "3.0") ;;
        sysadmin_ring:ments) FINAL_PARAMS=("0.25" "0.5") ;;
        sysadmin_ring:gs_power_uct) FINAL_PARAMS=("0.5" "2.5") ;;
        sysadmin_ring:gs_power_uct_f) FINAL_PARAMS=("0.75" "2.5") ;;

        *)
            echo "Missing fixed parameters for ${env_name} ${algorithm}" >&2
            return 1
            ;;
    esac
}
