#include <envs/passenger_grid.h>
#include <util.h>

#include <iostream>
#include <stdexcept>

using namespace std;

namespace poweruct {

    namespace {
        static const uint32_t LEFT = 0;
        static const uint32_t DOWN = 1;
        static const uint32_t RIGHT = 2;
        static const uint32_t UP = 3;

        static const char *action_names[] = {"LEFT", "DOWN", "RIGHT", "UP"};

        uint32_t turn_left(uint32_t action) {
            return (action + 3) % 4;
        }

        uint32_t turn_right(uint32_t action) {
            return (action + 1) % 4;
        }
    }

    PassengerGrid::PassengerGrid(uint32_t time_limit, bool slippery)
            : slippery(slippery), time_limit(time_limit), initialized(false), last_action(0), state(), initial_state(),
              start_position(0, 0), goal_position(width - 1, 0),
              // Edit these coordinates to move the three passengers around the map.
              passenger_positions{{GridPosition(1, 2), GridPosition(0, 5), GridPosition(6, 4)}},
              uniform_action_probs(num_actions, 1. / static_cast<double>(num_actions)), r_util() {
        if (time_limit == 0) {
            throw runtime_error("PassengerGrid requires a positive time limit");
        }

        initial_state = PassengerGridState(start_position.x, start_position.y, 0, 0);
        state = initial_state;
    }

    PassengerGrid::~PassengerGrid() = default;

    PassengerGridState PassengerGrid::getInitialState() {
        return initial_state;
    }

    uint32_t PassengerGrid::getInitialObservation() {
        return encode_observation(initial_state);
    }

    uint32_t PassengerGrid::getNumberOfObservations() {
        return width * height * (1U << num_passengers) * (time_limit + 1);
    }

    uint32_t PassengerGrid::getNumberOfActions() {
        return num_actions;
    }

    void PassengerGrid::seed(uint32_t s) {
        r_util.seed(s);
    }

    void PassengerGrid::reset() {
        state = initial_state;
        last_action = 0;
        initialized = true;
    }

    void PassengerGrid::set_state(PassengerGridState state) {
        validate_state(state);
        this->state = state;
        initialized = true;
    }

    std::tuple<PassengerGridState, uint32_t, double, bool> PassengerGrid::step(uint32_t action) {
        if (!initialized) {
            throw runtime_error(
                    "Environment has not been initialized! Call reset() or set_state() at least once before calling step()");
        }

        if (action >= num_actions) {
            throw runtime_error("Invalid action for PassengerGrid");
        }

        if (state.time >= time_limit) {
            throw runtime_error("Time Limit execeeded!");
        }

        uint32_t effective_action = sample_effective_action(action);
        last_action = effective_action;

        PassengerGridState next_state = state;
        next_state.time += 1;
        apply_action(next_state, effective_action);
        pickup_passenger(next_state);

        double reward = 0.;
        bool done = false;
        if (is_goal(next_state)) {
            uint32_t picked = count_picked_passengers(next_state.passenger_mask);
            static const double goal_rewards[] = {0., 1., 3., 7.};
            reward = goal_rewards[picked];
            done = true;
        } else if (next_state.time == time_limit) {
            done = true;
        }

        state = next_state;
        return make_tuple(state, encode_observation(state), reward, done);
    }

    std::tuple<uint32_t, PassengerGridState, uint32_t, double, bool> PassengerGrid::random_step() {
        uint32_t action = r_util.sample_discrete(uniform_action_probs);
        auto res = step(action);
        return make_tuple(action, get<0>(res), get<1>(res), get<2>(res), get<3>(res));
    }

    std::tuple<PassengerGridState, uint32_t, double, bool> PassengerGrid::simulate(PassengerGridState state,
                                                                                    uint32_t action) {
        set_state(state);
        return step(action);
    }

    std::tuple<uint32_t, PassengerGridState, uint32_t, double, bool>
    PassengerGrid::random_simulate(PassengerGridState state) {
        set_state(state);
        return random_step();
    }

    std::unique_ptr<std::vector<std::tuple<uint32_t, double, PassengerGridState, uint32_t >>>
    PassengerGrid::rollout(PassengerGridState state) {
        set_state(state);

        auto res = make_unique<std::vector<std::tuple<uint32_t, double, PassengerGridState, uint32_t >>>();
        bool done = false;
        while (!done) {
            uint32_t action = r_util.sample_discrete(uniform_action_probs);
            PassengerGridState next_state;
            uint32_t observation;
            double reward;
            tie(next_state, observation, reward, done) = step(action);
            res->emplace_back(make_tuple(action, reward, next_state, observation));
        }

        return res;
    }

    void PassengerGrid::render() {
        if (state.time > 0) {
            cout << "Action: " << action_names[last_action] << endl;
        }

        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                if (x == state.x && y == state.y) {
                    cout << 'A';
                } else {
                    cout << get_cell_char(x, y);
                }
            }
            cout << endl;
        }

        cout << "Picked passengers mask: " << state.passenger_mask << endl;
        cout << "Time: " << state.time << "/" << time_limit << endl;
        cout << endl;
    }

    uint32_t PassengerGrid::encode_observation(const PassengerGridState &state) const {
        validate_state(state);
        uint32_t spatial_id = state.y * width + state.x;
        return (((spatial_id << num_passengers) | state.passenger_mask) * (time_limit + 1)) + state.time;
    }

    uint32_t PassengerGrid::sample_effective_action(uint32_t action) {
        if (!slippery) {
            return action;
        }

        static vector<double> slip_probabilities({1. / 4., 1. / 2., 1. / 4.});
        static vector<uint32_t> candidate_actions(3);
        candidate_actions[0] = turn_left(action);
        candidate_actions[1] = action;
        candidate_actions[2] = turn_right(action);
        return candidate_actions[r_util.sample_discrete(slip_probabilities)];
    }

    void PassengerGrid::apply_action(PassengerGridState &state, uint32_t action) const {
        uint32_t next_x = state.x;
        uint32_t next_y = state.y;

        if (action == LEFT) {
            if (next_x > 0) {
                next_x -= 1;
            }
        } else if (action == DOWN) {
            if (next_y + 1 < height) {
                next_y += 1;
            }
        } else if (action == RIGHT) {
            if (next_x + 1 < width) {
                next_x += 1;
            }
        } else if (action == UP) {
            if (next_y > 0) {
                next_y -= 1;
            }
        } else {
            throw runtime_error("Invalid direction in PassengerGrid");
        }

        state.x = next_x;
        state.y = next_y;
    }

    void PassengerGrid::pickup_passenger(PassengerGridState &state) const {
        for (uint32_t i = 0; i < num_passengers; i++) {
            const auto &passenger_position = passenger_positions[i];
            if (passenger_position.x == state.x && passenger_position.y == state.y) {
                state.passenger_mask |= (1U << i);
            }
        }
    }

    bool PassengerGrid::is_goal(const PassengerGridState &state) const {
        return state.x == goal_position.x && state.y == goal_position.y;
    }

    uint32_t PassengerGrid::count_picked_passengers(uint32_t passenger_mask) const {
        uint32_t count = 0;
        for (uint32_t i = 0; i < num_passengers; i++) {
            if ((passenger_mask & (1U << i)) != 0U) {
                count += 1;
            }
        }
        return count;
    }

    bool PassengerGrid::is_inside(uint32_t x, uint32_t y) const {
        return x < width && y < height;
    }

    char PassengerGrid::get_cell_char(uint32_t x, uint32_t y) const {
        if (x == start_position.x && y == start_position.y) {
            return 'S';
        }

        if (x == goal_position.x && y == goal_position.y) {
            return 'G';
        }

        for (uint32_t i = 0; i < num_passengers; i++) {
            if ((state.passenger_mask & (1U << i)) == 0U &&
                passenger_positions[i].x == x && passenger_positions[i].y == y) {
                return static_cast<char>('1' + i);
            }
        }

        return '.';
    }

    void PassengerGrid::validate_state(const PassengerGridState &state) const {
        if (!is_inside(state.x, state.y)) {
            throw runtime_error("PassengerGrid state is outside the map");
        }

        if (state.passenger_mask >= (1U << num_passengers)) {
            throw runtime_error("PassengerGrid passenger mask is invalid");
        }

        if (state.time > time_limit) {
            throw runtime_error("PassengerGrid state exceeds the time limit");
        }
    }

}
