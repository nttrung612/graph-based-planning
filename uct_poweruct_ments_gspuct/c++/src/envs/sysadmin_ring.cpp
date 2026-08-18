#include <envs/sysadmin_ring.h>
#include <util.h>

#include <iostream>
#include <limits>
#include <string>
#include <stdexcept>

using namespace std;

namespace poweruct {

    namespace {
        constexpr double BOTH_CRASHED_RUNNING_PROBABILITY = 0.0238;
        constexpr double PREV_RUNNING_SELF_CRASHED_RUNNING_PROBABILITY = 0.0475;
        constexpr double PREV_CRASHED_SELF_RUNNING_PROBABILITY = 0.525;
        constexpr double BOTH_RUNNING_RUNNING_PROBABILITY = 0.95;

        uint64_t checked_power_of_two(uint32_t exponent) {
            return uint64_t(1) << exponent;
        }

        uint32_t alive_mask_limit() {
            return (uint32_t(1) << SYSADMIN_RING_NUM_COMPUTERS) - 1U;
        }

        string format_action(uint32_t action) {
            if (action < SYSADMIN_RING_NUM_COMPUTERS) {
                return string("REBOOT[") + to_string(action) + "]";
            }

            if (action == SYSADMIN_RING_NUM_COMPUTERS) {
                return "IDLE";
            }

            return string("UNKNOWN(") + to_string(action) + ")";
        }
    }

    SysAdminRing::SysAdminRing()
            : initialized(false), last_action(SYSADMIN_RING_NUM_COMPUTERS), num_observations(0), state(),
              initial_state(0, 0), r_util(),
              uniform_action_probs(SYSADMIN_RING_NUM_COMPUTERS + 1,
                                   1. / static_cast<double>(SYSADMIN_RING_NUM_COMPUTERS + 1)) {
        if (SYSADMIN_RING_NUM_COMPUTERS == 0 || SYSADMIN_RING_NUM_COMPUTERS > 27) {
            throw runtime_error("SysAdminRing requires SYSADMIN_RING_NUM_COMPUTERS in [1, 27]");
        }
        if (SYSADMIN_RING_TIME_LIMIT == 0) {
            throw runtime_error("SysAdminRing requires SYSADMIN_RING_TIME_LIMIT > 0");
        }

        uint64_t state_count = checked_power_of_two(SYSADMIN_RING_NUM_COMPUTERS);
        uint64_t observation_count = state_count * static_cast<uint64_t>(SYSADMIN_RING_TIME_LIMIT + 1);
        if (observation_count > static_cast<uint64_t>(numeric_limits<uint32_t>::max())) {
            throw runtime_error("SysAdminRing observation space does not fit into uint32_t");
        }

        num_observations = static_cast<uint32_t>(observation_count);
        state = initial_state;
    }

    SysAdminRing::~SysAdminRing() = default;

    SysAdminRingState SysAdminRing::getInitialState() {
        return initial_state;
    }

    uint32_t SysAdminRing::getInitialObservation() {
        return encode_observation(initial_state);
    }

    uint32_t SysAdminRing::getNumberOfObservations() {
        return num_observations;
    }

    uint32_t SysAdminRing::getNumberOfActions() {
        return SYSADMIN_RING_NUM_COMPUTERS + 1;
    }

    void SysAdminRing::seed(uint32_t s) {
        r_util.seed(s);
    }

    void SysAdminRing::reset() {
        state = initial_state;
        last_action = SYSADMIN_RING_NUM_COMPUTERS;
        initialized = true;
    }

    void SysAdminRing::set_state(SysAdminRingState state) {
        validate_state(state);
        this->state = state;
        initialized = true;
    }

    std::tuple<SysAdminRingState, uint32_t, double, bool> SysAdminRing::step(uint32_t action) {
        if (!initialized) {
            throw runtime_error(
                    "Environment has not been initialized! Call reset() or set_state() at least once before calling step()");
        }
        if (action >= getNumberOfActions()) {
            throw runtime_error("Invalid action for SysAdminRing");
        }
        if (state.time >= SYSADMIN_RING_TIME_LIMIT) {
            throw runtime_error("Time Limit execeeded!");
        }

        uint32_t next_alive_mask = 0;
        for (uint32_t machine = 0; machine < SYSADMIN_RING_NUM_COMPUTERS; machine++) {
            bool next_running = false;
            if (action == machine) {
                next_running = true;
            } else {
                uint32_t prev_machine = (machine + SYSADMIN_RING_NUM_COMPUTERS - 1) % SYSADMIN_RING_NUM_COMPUTERS;
                bool prev_running = is_running(state.alive_mask, prev_machine);
                bool self_running = is_running(state.alive_mask, machine);
                double probability = running_probability(prev_running, self_running);
                next_running = r_util.sample_uniform() < probability;
            }

            if (next_running) {
                next_alive_mask |= (uint32_t(1) << machine);
            }
        }

        SysAdminRingState next_state(next_alive_mask, state.time + 1);
        double reward = static_cast<double>(count_running(next_state.alive_mask)) /
                        static_cast<double>(SYSADMIN_RING_NUM_COMPUTERS);
        bool done = next_state.time == SYSADMIN_RING_TIME_LIMIT;

        state = next_state;
        last_action = action;
        return make_tuple(state, encode_observation(state), reward, done);
    }

    std::tuple<uint32_t, SysAdminRingState, uint32_t, double, bool> SysAdminRing::random_step() {
        uint32_t action = r_util.sample_discrete(uniform_action_probs);
        auto res = step(action);
        return make_tuple(action, get<0>(res), get<1>(res), get<2>(res), get<3>(res));
    }

    std::tuple<SysAdminRingState, uint32_t, double, bool> SysAdminRing::simulate(SysAdminRingState state,
                                                                                  uint32_t action) {
        set_state(state);
        return step(action);
    }

    std::tuple<uint32_t, SysAdminRingState, uint32_t, double, bool>
    SysAdminRing::random_simulate(SysAdminRingState state) {
        set_state(state);
        return random_step();
    }

    std::unique_ptr<std::vector<std::tuple<uint32_t, double, SysAdminRingState, uint32_t >>>
    SysAdminRing::rollout(SysAdminRingState state) {
        set_state(state);

        auto res = make_unique<std::vector<std::tuple<uint32_t, double, SysAdminRingState, uint32_t >>>();
        bool done = false;
        while (!done) {
            uint32_t action;
            SysAdminRingState next_state;
            uint32_t observation;
            double reward;
            tie(action, next_state, observation, reward, done) = random_step();
            res->emplace_back(make_tuple(action, reward, next_state, observation));
        }

        return res;
    }

    void SysAdminRing::render() {
        if (state.time > 0) {
            cout << "Action: " << format_action(last_action) << endl;
        }

        cout << "Machines: [";
        for (uint32_t machine = 0; machine < SYSADMIN_RING_NUM_COMPUTERS; machine++) {
            if (machine > 0) {
                cout << ' ';
            }
            cout << (is_running(state.alive_mask, machine) ? '1' : '0');
        }
        cout << "]" << endl;
        cout << "Running: " << count_running(state.alive_mask) << "/" << SYSADMIN_RING_NUM_COMPUTERS << endl;
        cout << "Time: " << state.time << "/" << SYSADMIN_RING_TIME_LIMIT << endl;
        cout << endl;
    }

    uint32_t SysAdminRing::encode_observation(const SysAdminRingState &state) const {
        validate_state(state);

        uint64_t observation =
                static_cast<uint64_t>(state.alive_mask) * static_cast<uint64_t>(SYSADMIN_RING_TIME_LIMIT + 1) +
                static_cast<uint64_t>(state.time);
        if (observation > static_cast<uint64_t>(numeric_limits<uint32_t>::max())) {
            throw runtime_error("SysAdminRing observation index does not fit into uint32_t");
        }

        return static_cast<uint32_t>(observation);
    }

    void SysAdminRing::validate_state(const SysAdminRingState &state) const {
        if ((state.alive_mask & ~alive_mask_limit()) != 0U) {
            throw runtime_error("SysAdminRing alive mask contains invalid bits");
        }
        if (state.time > SYSADMIN_RING_TIME_LIMIT) {
            throw runtime_error("SysAdminRing state exceeds the time limit");
        }
    }

    bool SysAdminRing::is_running(uint32_t alive_mask, uint32_t machine) const {
        return (alive_mask & (uint32_t(1) << machine)) != 0U;
    }

    uint32_t SysAdminRing::count_running(uint32_t alive_mask) const {
        uint32_t count = 0;
        for (uint32_t machine = 0; machine < SYSADMIN_RING_NUM_COMPUTERS; machine++) {
            if (is_running(alive_mask, machine)) {
                count += 1;
            }
        }
        return count;
    }

    double SysAdminRing::running_probability(bool prev_running, bool self_running) const {
        if (!prev_running && !self_running) {
            return BOTH_CRASHED_RUNNING_PROBABILITY;
        }
        if (prev_running && !self_running) {
            return PREV_RUNNING_SELF_CRASHED_RUNNING_PROBABILITY;
        }
        if (!prev_running && self_running) {
            return PREV_CRASHED_SELF_RUNNING_PROBABILITY;
        }

        return BOTH_RUNNING_RUNNING_PROBABILITY;
    }

}
