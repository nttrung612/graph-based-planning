#ifndef C___PASSENGER_GRID_H
#define C___PASSENGER_GRID_H

#include <environment.h>
#include <random_utils.h>

#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace poweruct {

    struct PassengerGridState {
        PassengerGridState() : x(0), y(0), passenger_mask(0), time(0) {}

        PassengerGridState(uint32_t x, uint32_t y, uint32_t passenger_mask, uint32_t time)
                : x(x), y(y), passenger_mask(passenger_mask), time(time) {}

        uint32_t x;
        uint32_t y;
        uint32_t passenger_mask;
        uint32_t time;
    };

    class PassengerGrid : public Environment<PassengerGridState> {

    public:
        PassengerGrid(uint32_t time_limit, bool slippery);

        ~PassengerGrid() override;

        PassengerGridState getInitialState() override;

        uint32_t getInitialObservation() override;

        uint32_t getNumberOfObservations() override;

        uint32_t getNumberOfActions() override;

        void seed(uint32_t s) override;

        void reset() override;

        void set_state(PassengerGridState state) override;

        std::tuple<PassengerGridState, uint32_t, double, bool> step(uint32_t action) override;

        std::tuple<uint32_t, PassengerGridState, uint32_t, double, bool> random_step() override;

        std::tuple<PassengerGridState, uint32_t, double, bool> simulate(PassengerGridState state,
                                                                         uint32_t action) override;

        std::tuple<uint32_t, PassengerGridState, uint32_t, double, bool>
        random_simulate(PassengerGridState state) override;

        std::unique_ptr<std::vector<std::tuple<uint32_t, double, PassengerGridState, uint32_t >>>
        rollout(PassengerGridState state) override;

        void render() override;

    private:
        struct GridPosition {
            GridPosition() : x(0), y(0) {}

            GridPosition(uint32_t x, uint32_t y) : x(x), y(y) {}

            uint32_t x;
            uint32_t y;
        };

        static constexpr uint32_t width = 7;
        static constexpr uint32_t height = 6;
        static constexpr uint32_t num_actions = 4;
        static constexpr uint32_t num_passengers = 3;

        bool slippery;
        uint32_t time_limit;
        bool initialized;
        uint32_t last_action;
        PassengerGridState state;
        PassengerGridState initial_state;
        GridPosition start_position;
        GridPosition goal_position;
        std::array<GridPosition, num_passengers> passenger_positions;
        std::vector<double> uniform_action_probs;
        RandomUtils r_util;

        uint32_t encode_observation(const PassengerGridState &state) const;

        uint32_t sample_effective_action(uint32_t action);

        void apply_action(PassengerGridState &state, uint32_t action) const;

        void pickup_passenger(PassengerGridState &state) const;

        bool is_goal(const PassengerGridState &state) const;

        uint32_t count_picked_passengers(uint32_t passenger_mask) const;

        bool is_inside(uint32_t x, uint32_t y) const;

        char get_cell_char(uint32_t x, uint32_t y) const;

        void validate_state(const PassengerGridState &state) const;
    };

}

#endif //C___PASSENGER_GRID_H
