#ifndef C___GRAPH_SEARCH_H
#define C___GRAPH_SEARCH_H

#include <tree_search.h>
#include <discrete_environment.h>
#include <envs/passenger_grid.h>
#include <envs/factored_river_swim.h>
#include <envs/four_rooms.h>
#include <envs/sysadmin_ring.h>

#include <array>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <tuple>

namespace poweruct {

    enum class GraphSearchMode {
        Depth,
        Full
    };

    struct GraphStateKey {
        uint32_t key_type = 0;
        uint32_t x = 0;
        uint32_t y = 0;
        uint32_t passenger_mask = 0;
        uint32_t time = 0;
        uint32_t factored_count = 0;
        std::array<uint32_t, FACTORED_RIVER_SWIM_NUM_RIVERS> factored_positions{};

        bool operator<(const GraphStateKey &other) const {
            return std::tie(key_type, x, y, passenger_mask, time, factored_count, factored_positions) <
                   std::tie(other.key_type, other.x, other.y, other.passenger_mask, other.time,
                            other.factored_count, other.factored_positions);
        }

        bool operator==(const GraphStateKey &other) const {
            return key_type == other.key_type &&
                   x == other.x &&
                   y == other.y &&
                   passenger_mask == other.passenger_mask &&
                   time == other.time &&
                   factored_count == other.factored_count &&
                   factored_positions == other.factored_positions;
        }
    };

    struct GraphStateKeyHash {
        std::size_t operator()(const GraphStateKey &key) const {
            std::size_t seed = 0;
            combine(seed, key.key_type);
            combine(seed, key.x);
            combine(seed, key.y);
            combine(seed, key.passenger_mask);
            combine(seed, key.time);
            combine(seed, key.factored_count);
            for (uint32_t position : key.factored_positions) {
                combine(seed, position);
            }
            return seed;
        }

    private:
        static void combine(std::size_t &seed, uint32_t value) {
            seed ^= std::hash<uint32_t>()(value) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        }
    };

    template<typename S>
    struct GraphStateTraits;

    template<>
    struct GraphStateTraits<DiscreteEnvironmentState> {
        static GraphStateKey build_key(const DiscreteEnvironmentState &state, GraphSearchMode mode) {
            GraphStateKey key;
            key.x = state.state % 8;
            key.y = state.state / 8;
            key.passenger_mask = 0;
            key.time = (mode == GraphSearchMode::Depth) ? state.time : 0;
            return key;
        }

    };

    template<>
    struct GraphStateTraits<PassengerGridState> {
        static GraphStateKey build_key(const PassengerGridState &state, GraphSearchMode mode) {
            GraphStateKey key;
            key.x = state.x;
            key.y = state.y;
            key.passenger_mask = state.passenger_mask;
            key.time = (mode == GraphSearchMode::Depth) ? state.time : 0;
            return key;
        }

    };

    template<>
    struct GraphStateTraits<FactoredRiverSwimState> {
        static GraphStateKey build_key(const FactoredRiverSwimState &state, GraphSearchMode mode) {
            GraphStateKey key;
            key.key_type = 1;
            key.factored_count = FACTORED_RIVER_SWIM_NUM_RIVERS;
            key.factored_positions = state.positions;
            key.time = (mode == GraphSearchMode::Depth) ? state.time : 0;
            return key;
        }
    };

    template<>
    struct GraphStateTraits<FourRoomsState> {
        static GraphStateKey build_key(const FourRoomsState &state, GraphSearchMode mode) {
            GraphStateKey key;
            key.key_type = 3;
            key.x = static_cast<uint32_t>(static_cast<int32_t>(state.x) -
                                          static_cast<int32_t>(state.start_x) +
                                          static_cast<int32_t>(FOUR_ROOMS_GRID_SIZE - 1));
            key.y = static_cast<uint32_t>(static_cast<int32_t>(state.y) -
                                          static_cast<int32_t>(state.start_y) +
                                          static_cast<int32_t>(FOUR_ROOMS_GRID_SIZE - 1));
            key.time = (mode == GraphSearchMode::Depth) ? state.time : 0;
            return key;
        }
    };

    template<>
    struct GraphStateTraits<SysAdminRingState> {
        static GraphStateKey build_key(const SysAdminRingState &state, GraphSearchMode mode) {
            GraphStateKey key;
            key.key_type = 2;
            key.x = state.alive_mask;
            key.y = SYSADMIN_RING_NUM_COMPUTERS;
            key.time = (mode == GraphSearchMode::Depth) ? state.time : 0;
            return key;
        }
    };

    template<typename S>
    inline GraphStateKey build_graph_state_key(const S &state, GraphSearchMode mode, const Environment<S> *env) {
        (void) env;
        return GraphStateTraits<S>::build_key(state, mode);
    }

    struct GraphActionStats {
        uint32_t action = 0;
        double q_value = 0.;
        uint32_t visits = 0;
    };

}

#endif //C___GRAPH_SEARCH_H
