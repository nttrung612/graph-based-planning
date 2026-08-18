#include <envs/factored_river_swim.h>
#include <util.h>

#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace poweruct {

    namespace {
        constexpr double START_REST_REWARD = 0.1;
        constexpr double GOAL_SWIM_REWARD = 1.0;

        uint64_t checked_power(uint32_t base, uint32_t exponent) {
            uint64_t result = 1;
            for (uint32_t i = 0; i < exponent; i++) {
                result *= static_cast<uint64_t>(base);
            }
            return result;
        }
    }

    FactoredRiverSwim::FactoredRiverSwim()
            : initialized(false), last_action(0), num_observations(0), state(), initial_state(), r_util(),
              uniform_action_probs(1U << FACTORED_RIVER_SWIM_NUM_RIVERS,
                                   1. / static_cast<double>(1U << FACTORED_RIVER_SWIM_NUM_RIVERS)) {
        if (FACTORED_RIVER_SWIM_NUM_RIVERS == 0 || FACTORED_RIVER_SWIM_NUM_RIVERS > 5) {
            throw runtime_error("FactoredRiverSwim requires FACTORED_RIVER_SWIM_NUM_RIVERS in [1, 5]");
        }
        if (FACTORED_RIVER_SWIM_NUM_LOCATIONS < 2) {
            throw runtime_error("FactoredRiverSwim requires FACTORED_RIVER_SWIM_NUM_LOCATIONS >= 2");
        }
        if (FACTORED_RIVER_SWIM_TIME_LIMIT == 0) {
            throw runtime_error("FactoredRiverSwim requires FACTORED_RIVER_SWIM_TIME_LIMIT > 0");
        }

        uint64_t spatial_state_count = checked_power(FACTORED_RIVER_SWIM_NUM_LOCATIONS, FACTORED_RIVER_SWIM_NUM_RIVERS);
        uint64_t total_observation_count =
                spatial_state_count * static_cast<uint64_t>(FACTORED_RIVER_SWIM_TIME_LIMIT + 1);
        if (total_observation_count > static_cast<uint64_t>(numeric_limits<uint32_t>::max())) {
            throw runtime_error("FactoredRiverSwim observation space does not fit into uint32_t");
        }

        num_observations = static_cast<uint32_t>(total_observation_count);
        state = initial_state;
    }

    FactoredRiverSwim::~FactoredRiverSwim() = default;

    FactoredRiverSwimState FactoredRiverSwim::getInitialState() {
        return initial_state;
    }

    uint32_t FactoredRiverSwim::getInitialObservation() {
        return encode_observation(initial_state);
    }

    uint32_t FactoredRiverSwim::getNumberOfObservations() {
        return num_observations;
    }

    uint32_t FactoredRiverSwim::getNumberOfActions() {
        return static_cast<uint32_t>(uniform_action_probs.size());
    }

    void FactoredRiverSwim::seed(uint32_t s) {
        r_util.seed(s);
    }

    void FactoredRiverSwim::reset() {
        state = initial_state;
        last_action = 0;
        initialized = true;
    }

    void FactoredRiverSwim::set_state(FactoredRiverSwimState state) {
        validate_state(state);
        this->state = state;
        initialized = true;
    }

    std::tuple<FactoredRiverSwimState, uint32_t, double, bool> FactoredRiverSwim::step(uint32_t action) {
        if (!initialized) {
            throw runtime_error(
                    "Environment has not been initialized! Call reset() or set_state() at least once before calling step()");
        }
        if (action >= getNumberOfActions()) {
            throw runtime_error("Invalid action for FactoredRiverSwim");
        }
        if (state.time >= FACTORED_RIVER_SWIM_TIME_LIMIT) {
            throw runtime_error("Time Limit execeeded!");
        }

        auto action_bits = decode_action(action);
        double reward = compute_reward(state, action_bits);

        FactoredRiverSwimState next_state = state;
        next_state.time += 1;
        for (uint32_t river = 0; river < FACTORED_RIVER_SWIM_NUM_RIVERS; river++) {
            next_state.positions[river] = sample_next_position(state.positions[river], action_bits[river]);
        }

        state = next_state;
        last_action = action;
        bool done = state.time == FACTORED_RIVER_SWIM_TIME_LIMIT;
        return make_tuple(state, encode_observation(state), reward, done);
    }

    std::tuple<uint32_t, FactoredRiverSwimState, uint32_t, double, bool> FactoredRiverSwim::random_step() {
        uint32_t action = r_util.sample_discrete(uniform_action_probs);
        auto res = step(action);
        return make_tuple(action, get<0>(res), get<1>(res), get<2>(res), get<3>(res));
    }

    std::tuple<FactoredRiverSwimState, uint32_t, double, bool>
    FactoredRiverSwim::simulate(FactoredRiverSwimState state, uint32_t action) {
        set_state(state);
        return step(action);
    }

    std::tuple<uint32_t, FactoredRiverSwimState, uint32_t, double, bool>
    FactoredRiverSwim::random_simulate(FactoredRiverSwimState state) {
        set_state(state);
        return random_step();
    }

    std::unique_ptr<std::vector<std::tuple<uint32_t, double, FactoredRiverSwimState, uint32_t >>>
    FactoredRiverSwim::rollout(FactoredRiverSwimState state) {
        set_state(state);

        auto res = make_unique<std::vector<std::tuple<uint32_t, double, FactoredRiverSwimState, uint32_t >>>();
        bool done = false;
        while (!done) {
            uint32_t action;
            FactoredRiverSwimState next_state;
            uint32_t observation;
            double reward;
            tie(action, next_state, observation, reward, done) = random_step();
            res->emplace_back(make_tuple(action, reward, next_state, observation));
        }

        return res;
    }

    void FactoredRiverSwim::render() {
        if (state.time > 0) {
            cout << "Action: " << format_action(last_action) << endl;
        }

        cout << "Positions: [";
        for (uint32_t river = 0; river < FACTORED_RIVER_SWIM_NUM_RIVERS; river++) {
            if (river > 0) {
                cout << ", ";
            }
            cout << state.positions[river];
        }
        cout << "]" << endl;
        cout << "Goal position: " << (FACTORED_RIVER_SWIM_NUM_LOCATIONS - 1) << endl;
        cout << "Time: " << state.time << "/" << FACTORED_RIVER_SWIM_TIME_LIMIT << endl;
        cout << endl;
    }

    uint32_t FactoredRiverSwim::encode_observation(const FactoredRiverSwimState &state) const {
        validate_state(state);

        uint64_t spatial_id = 0;
        uint64_t factor = 1;
        for (uint32_t river = 0; river < FACTORED_RIVER_SWIM_NUM_RIVERS; river++) {
            spatial_id += static_cast<uint64_t>(state.positions[river]) * factor;
            factor *= static_cast<uint64_t>(FACTORED_RIVER_SWIM_NUM_LOCATIONS);
        }

        uint64_t observation =
                spatial_id * static_cast<uint64_t>(FACTORED_RIVER_SWIM_TIME_LIMIT + 1) + state.time;
        if (observation > static_cast<uint64_t>(numeric_limits<uint32_t>::max())) {
            throw runtime_error("FactoredRiverSwim observation index does not fit into uint32_t");
        }

        return static_cast<uint32_t>(observation);
    }

    void FactoredRiverSwim::validate_state(const FactoredRiverSwimState &state) const {
        if (state.time > FACTORED_RIVER_SWIM_TIME_LIMIT) {
            throw runtime_error("FactoredRiverSwim state exceeds the time limit");
        }

        for (uint32_t river = 0; river < FACTORED_RIVER_SWIM_NUM_RIVERS; river++) {
            if (state.positions[river] >= FACTORED_RIVER_SWIM_NUM_LOCATIONS) {
                throw runtime_error("FactoredRiverSwim position is outside the river");
            }
        }
    }

    std::array<uint32_t, FACTORED_RIVER_SWIM_NUM_RIVERS> FactoredRiverSwim::decode_action(uint32_t action) const {
        std::array<uint32_t, FACTORED_RIVER_SWIM_NUM_RIVERS> action_bits{};
        for (uint32_t river = 0; river < FACTORED_RIVER_SWIM_NUM_RIVERS; river++) {
            action_bits[river] = (action >> river) & 1U;
        }
        return action_bits;
    }

    uint32_t FactoredRiverSwim::sample_next_position(uint32_t position, uint32_t action_bit) {
        uint32_t goal_position = FACTORED_RIVER_SWIM_NUM_LOCATIONS - 1;
        if (action_bit == 0) {
            return position == 0 ? 0 : position - 1;
        }

        double sample = r_util.sample_uniform();
        if (position == 0) {
            return sample < 0.4 ? 0 : 1;
        }
        if (position == goal_position) {
            return sample < 0.4 ? goal_position - 1 : goal_position;
        }
        if (sample < 0.05) {
            return position - 1;
        }
        if (sample < 0.65) {
            return position;
        }
        return position + 1;
    }

    double FactoredRiverSwim::compute_reward(
            const FactoredRiverSwimState &state,
            const std::array<uint32_t, FACTORED_RIVER_SWIM_NUM_RIVERS> &action_bits) const {
        double reward = 0.;
        bool all_at_goal = true;
        bool all_swim_up = true;
        uint32_t goal_position = FACTORED_RIVER_SWIM_NUM_LOCATIONS - 1;

        for (uint32_t river = 0; river < FACTORED_RIVER_SWIM_NUM_RIVERS; river++) {
            uint32_t position = state.positions[river];
            uint32_t action_bit = action_bits[river];

            if (position == 0 && action_bit == 0) {
                reward += START_REST_REWARD;
            }
            if (position == goal_position && action_bit == 1) {
                reward += GOAL_SWIM_REWARD;
            }

            all_at_goal = all_at_goal && (position == goal_position);
            all_swim_up = all_swim_up && (action_bit == 1);
        }

        if (all_at_goal && all_swim_up) {
            reward += static_cast<double>(FACTORED_RIVER_SWIM_NUM_RIVERS);
        }

        return reward / (2. * static_cast<double>(FACTORED_RIVER_SWIM_NUM_RIVERS));
    }

    std::string FactoredRiverSwim::format_action(uint32_t action) const {
        if (action >= static_cast<uint32_t>(uniform_action_probs.size())) {
            return string("UNKNOWN(") + to_string(action) + ")";
        }

        auto action_bits = decode_action(action);
        ostringstream out;
        out << "[";
        for (uint32_t river = 0; river < FACTORED_RIVER_SWIM_NUM_RIVERS; river++) {
            if (river > 0) {
                out << ", ";
            }
            out << action_bits[river];
        }
        out << "]";
        return out.str();
    }

}
