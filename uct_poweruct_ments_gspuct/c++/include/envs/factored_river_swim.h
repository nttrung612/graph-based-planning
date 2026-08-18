#ifndef C___FACTORED_RIVER_SWIM_H
#define C___FACTORED_RIVER_SWIM_H

#include <environment.h>
#include <random_utils.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace poweruct {

    // Edit these compile-time constants to change the fixed environment size.
    constexpr uint32_t FACTORED_RIVER_SWIM_NUM_RIVERS = 3;
    constexpr uint32_t FACTORED_RIVER_SWIM_NUM_LOCATIONS = 5;
    constexpr uint32_t FACTORED_RIVER_SWIM_TIME_LIMIT = 35;

    struct FactoredRiverSwimState {
        FactoredRiverSwimState() : positions(), time(0) {
            positions.fill(0);
        }

        FactoredRiverSwimState(const std::array<uint32_t, FACTORED_RIVER_SWIM_NUM_RIVERS> &positions, uint32_t time)
                : positions(positions), time(time) {}

        std::array<uint32_t, FACTORED_RIVER_SWIM_NUM_RIVERS> positions;
        uint32_t time;
    };

    class FactoredRiverSwim : public Environment<FactoredRiverSwimState> {

    public:
        FactoredRiverSwim();

        ~FactoredRiverSwim() override;

        FactoredRiverSwimState getInitialState() override;

        uint32_t getInitialObservation() override;

        uint32_t getNumberOfObservations() override;

        uint32_t getNumberOfActions() override;

        void seed(uint32_t s) override;

        void reset() override;

        void set_state(FactoredRiverSwimState state) override;

        std::tuple<FactoredRiverSwimState, uint32_t, double, bool> step(uint32_t action) override;

        std::tuple<uint32_t, FactoredRiverSwimState, uint32_t, double, bool> random_step() override;

        std::tuple<FactoredRiverSwimState, uint32_t, double, bool> simulate(FactoredRiverSwimState state,
                                                                             uint32_t action) override;

        std::tuple<uint32_t, FactoredRiverSwimState, uint32_t, double, bool>
        random_simulate(FactoredRiverSwimState state) override;

        std::unique_ptr<std::vector<std::tuple<uint32_t, double, FactoredRiverSwimState, uint32_t >>>
        rollout(FactoredRiverSwimState state) override;

        void render() override;

    private:
        bool initialized;
        uint32_t last_action;
        uint32_t num_observations;
        FactoredRiverSwimState state;
        FactoredRiverSwimState initial_state;
        RandomUtils r_util;
        std::vector<double> uniform_action_probs;

        uint32_t encode_observation(const FactoredRiverSwimState &state) const;

        void validate_state(const FactoredRiverSwimState &state) const;

        std::array<uint32_t, FACTORED_RIVER_SWIM_NUM_RIVERS> decode_action(uint32_t action) const;

        uint32_t sample_next_position(uint32_t position, uint32_t action_bit);

        double compute_reward(const FactoredRiverSwimState &state,
                              const std::array<uint32_t, FACTORED_RIVER_SWIM_NUM_RIVERS> &action_bits) const;

        std::string format_action(uint32_t action) const;
    };

}

#endif //C___FACTORED_RIVER_SWIM_H
