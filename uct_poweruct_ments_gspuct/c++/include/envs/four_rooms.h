#ifndef C___FOUR_ROOMS_H
#define C___FOUR_ROOMS_H

#include <environment.h>
#include <random_utils.h>

#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace poweruct {

    // Edit this compile-time constant to change the fixed environment size.
    constexpr uint32_t FOUR_ROOMS_N = 5;
    constexpr uint32_t FOUR_ROOMS_GRID_SIZE = 2 * FOUR_ROOMS_N + 1;
    constexpr uint32_t FOUR_ROOMS_TIME_LIMIT = 50;
    constexpr uint32_t FOUR_ROOMS_NUM_DOORS = 4;

    struct FourRoomsState {
        FourRoomsState() : x(0), y(0), start_x(0), start_y(0), goal_x(0), goal_y(0), time(0), door_cells() {
            door_cells.fill(0);
        }

        FourRoomsState(uint32_t x, uint32_t y, uint32_t start_x, uint32_t start_y,
                       uint32_t goal_x, uint32_t goal_y, uint32_t time,
                       const std::array<uint32_t, FOUR_ROOMS_NUM_DOORS> &door_cells)
                : x(x), y(y), start_x(start_x), start_y(start_y),
                  goal_x(goal_x), goal_y(goal_y), time(time), door_cells(door_cells) {}

        uint32_t x;
        uint32_t y;
        uint32_t start_x;
        uint32_t start_y;
        uint32_t goal_x;
        uint32_t goal_y;
        uint32_t time;
        std::array<uint32_t, FOUR_ROOMS_NUM_DOORS> door_cells;
    };

    class FourRooms : public Environment<FourRoomsState> {

    public:
        FourRooms();

        ~FourRooms() override;

        FourRoomsState getInitialState() override;

        uint32_t getInitialObservation() override;

        uint32_t getNumberOfObservations() override;

        uint32_t getNumberOfActions() override;

        void seed(uint32_t s) override;

        void reset() override;

        void set_state(FourRoomsState state) override;

        std::tuple<FourRoomsState, uint32_t, double, bool> step(uint32_t action) override;

        std::tuple<uint32_t, FourRoomsState, uint32_t, double, bool> random_step() override;

        std::tuple<FourRoomsState, uint32_t, double, bool> simulate(FourRoomsState state, uint32_t action) override;

        std::tuple<uint32_t, FourRoomsState, uint32_t, double, bool> random_simulate(FourRoomsState state) override;

        std::unique_ptr<std::vector<std::tuple<uint32_t, double, FourRoomsState, uint32_t >>>
        rollout(FourRoomsState state) override;

        void render() override;

    private:
        static constexpr uint32_t num_actions = 4;

        bool initialized;
        uint32_t last_action;
        FourRoomsState state;
        FourRoomsState initial_state;
        RandomUtils r_util;
        std::vector<double> uniform_action_probs;
        std::vector<double> slip_action_probs;

        uint32_t encode_observation(const FourRoomsState &state) const;

        FourRoomsState sample_initial_state();

        uint32_t sample_effective_action(uint32_t action);

        FourRoomsState apply_action(const FourRoomsState &state, uint32_t action) const;

        bool is_goal(const FourRoomsState &state) const;

        double compute_goal_reward(uint32_t step_count) const;

        bool is_inside(int32_t x, int32_t y) const;

        bool is_wall(uint32_t x, uint32_t y, const std::array<uint32_t, FOUR_ROOMS_NUM_DOORS> &door_cells) const;

        bool is_door_cell(uint32_t cell_id,
                          const std::array<uint32_t, FOUR_ROOMS_NUM_DOORS> &door_cells) const;

        bool is_valid_spawn_cell(uint32_t x, uint32_t y,
                                 const std::array<uint32_t, FOUR_ROOMS_NUM_DOORS> &door_cells) const;

        uint32_t encode_cell(uint32_t x, uint32_t y) const;

        std::pair<uint32_t, uint32_t> decode_cell(uint32_t cell_id) const;

        void validate_state(const FourRoomsState &state) const;

        bool is_valid_door_cell(uint32_t cell_id, uint32_t door_index) const;

        const char *format_action(uint32_t action) const;
    };

}

#endif //C___FOUR_ROOMS_H
