#include <envs/four_rooms.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace std;

namespace poweruct {

    namespace {
        static const uint32_t LEFT = 0;
        static const uint32_t DOWN = 1;
        static const uint32_t RIGHT = 2;
        static const uint32_t UP = 3;

        uint32_t turn_left(uint32_t action) {
            return (action + 3) % 4;
        }

        uint32_t turn_right(uint32_t action) {
            return (action + 1) % 4;
        }
    }

    FourRooms::FourRooms()
            : initialized(false), last_action(0), state(), initial_state(), r_util(),
              uniform_action_probs(num_actions, 1. / static_cast<double>(num_actions)),
              slip_action_probs({0.25, 0.5, 0.25}) {
        if (FOUR_ROOMS_N < 2) {
            throw runtime_error("FourRooms requires FOUR_ROOMS_N >= 2");
        }

        std::array<uint32_t, FOUR_ROOMS_NUM_DOORS> default_doors;
        default_doors[0] = encode_cell(FOUR_ROOMS_N, 0);
        default_doors[1] = encode_cell(FOUR_ROOMS_N, FOUR_ROOMS_N + 1);
        default_doors[2] = encode_cell(0, FOUR_ROOMS_N);
        default_doors[3] = encode_cell(FOUR_ROOMS_N + 1, FOUR_ROOMS_N);
        initial_state = FourRoomsState(0, 0, 0, 0,
                                       FOUR_ROOMS_GRID_SIZE - 1, FOUR_ROOMS_GRID_SIZE - 1, 0, default_doors);
        state = initial_state;
    }

    FourRooms::~FourRooms() = default;

    FourRoomsState FourRooms::getInitialState() {
        return initial_state;
    }

    uint32_t FourRooms::getInitialObservation() {
        return encode_observation(initial_state);
    }

    uint32_t FourRooms::getNumberOfObservations() {
        return FOUR_ROOMS_GRID_SIZE * FOUR_ROOMS_GRID_SIZE * (FOUR_ROOMS_TIME_LIMIT + 1);
    }

    uint32_t FourRooms::getNumberOfActions() {
        return num_actions;
    }

    void FourRooms::seed(uint32_t s) {
        r_util.seed(s);
    }

    void FourRooms::reset() {
        initial_state = sample_initial_state();
        state = initial_state;
        last_action = 0;
        initialized = true;
    }

    void FourRooms::set_state(FourRoomsState state) {
        validate_state(state);
        this->state = state;
        initialized = true;
    }

    std::tuple<FourRoomsState, uint32_t, double, bool> FourRooms::step(uint32_t action) {
        if (!initialized) {
            throw runtime_error(
                    "Environment has not been initialized! Call reset() or set_state() at least once before calling step()");
        }
        if (action >= num_actions) {
            throw runtime_error("Invalid action for FourRooms");
        }
        if (state.time >= FOUR_ROOMS_TIME_LIMIT) {
            throw runtime_error("Time Limit execeeded!");
        }

        uint32_t effective_action = sample_effective_action(action);
        last_action = effective_action;

        FourRoomsState next_state = apply_action(state, effective_action);
        next_state.time += 1;

        double reward = 0.;
        bool done = false;
        if (is_goal(next_state)) {
            reward = compute_goal_reward(next_state.time);
            done = true;
        } else if (next_state.time == FOUR_ROOMS_TIME_LIMIT) {
            done = true;
        }

        state = next_state;
        return make_tuple(state, encode_observation(state), reward, done);
    }

    std::tuple<uint32_t, FourRoomsState, uint32_t, double, bool> FourRooms::random_step() {
        uint32_t action = r_util.sample_discrete(uniform_action_probs);
        auto res = step(action);
        return make_tuple(action, get<0>(res), get<1>(res), get<2>(res), get<3>(res));
    }

    std::tuple<FourRoomsState, uint32_t, double, bool> FourRooms::simulate(FourRoomsState state, uint32_t action) {
        set_state(state);
        return step(action);
    }

    std::tuple<uint32_t, FourRoomsState, uint32_t, double, bool> FourRooms::random_simulate(FourRoomsState state) {
        set_state(state);
        return random_step();
    }

    std::unique_ptr<std::vector<std::tuple<uint32_t, double, FourRoomsState, uint32_t >>>
    FourRooms::rollout(FourRoomsState state) {
        set_state(state);

        auto res = make_unique<std::vector<std::tuple<uint32_t, double, FourRoomsState, uint32_t >>>();
        bool done = false;
        while (!done) {
            uint32_t action;
            FourRoomsState next_state;
            uint32_t observation;
            double reward;
            tie(action, next_state, observation, reward, done) = random_step();
            res->emplace_back(make_tuple(action, reward, next_state, observation));
        }

        return res;
    }

    void FourRooms::render() {
        if (state.time > 0) {
            cout << "Action: " << format_action(last_action) << endl;
        }

        for (uint32_t y = 0; y < FOUR_ROOMS_GRID_SIZE; y++) {
            for (uint32_t x = 0; x < FOUR_ROOMS_GRID_SIZE; x++) {
                uint32_t cell_id = encode_cell(x, y);
                char cell_char = '.';
                if (state.x == x && state.y == y) {
                    cell_char = 'A';
                } else if (state.goal_x == x && state.goal_y == y) {
                    cell_char = 'G';
                } else if (state.start_x == x && state.start_y == y) {
                    cell_char = 'S';
                } else if (is_door_cell(cell_id, state.door_cells)) {
                    cell_char = 'D';
                } else if (is_wall(x, y, state.door_cells)) {
                    cell_char = '#';
                }
                cout << cell_char;
            }
            cout << endl;
        }

        cout << "Time: " << state.time << "/" << FOUR_ROOMS_TIME_LIMIT << endl;
        cout << endl;
    }

    uint32_t FourRooms::encode_observation(const FourRoomsState &state) const {
        validate_state(state);
        return encode_cell(state.x, state.y) * (FOUR_ROOMS_TIME_LIMIT + 1) + state.time;
    }

    FourRoomsState FourRooms::sample_initial_state() {
        std::array<uint32_t, FOUR_ROOMS_NUM_DOORS> door_cells;
        door_cells[0] = encode_cell(FOUR_ROOMS_N, r_util.sample_uniform(FOUR_ROOMS_N));
        door_cells[1] = encode_cell(FOUR_ROOMS_N, FOUR_ROOMS_N + 1 + r_util.sample_uniform(FOUR_ROOMS_N));
        door_cells[2] = encode_cell(r_util.sample_uniform(FOUR_ROOMS_N), FOUR_ROOMS_N);
        door_cells[3] = encode_cell(FOUR_ROOMS_N + 1 + r_util.sample_uniform(FOUR_ROOMS_N), FOUR_ROOMS_N);

        vector<uint32_t> candidate_cells;
        candidate_cells.reserve(FOUR_ROOMS_GRID_SIZE * FOUR_ROOMS_GRID_SIZE);
        for (uint32_t y = 0; y < FOUR_ROOMS_GRID_SIZE; y++) {
            for (uint32_t x = 0; x < FOUR_ROOMS_GRID_SIZE; x++) {
                if (is_valid_spawn_cell(x, y, door_cells)) {
                    candidate_cells.emplace_back(encode_cell(x, y));
                }
            }
        }

        if (candidate_cells.size() < 2) {
            throw runtime_error("FourRooms requires at least two valid spawn cells");
        }

        uint32_t start_cell = r_util.sample_uniform(candidate_cells);
        uint32_t goal_cell = start_cell;
        while (goal_cell == start_cell) {
            goal_cell = r_util.sample_uniform(candidate_cells);
        }

        auto start = decode_cell(start_cell);
        auto goal = decode_cell(goal_cell);
        return FourRoomsState(start.first, start.second, start.first, start.second,
                              goal.first, goal.second, 0, door_cells);
    }

    uint32_t FourRooms::sample_effective_action(uint32_t action) {
        static vector<uint32_t> candidate_actions(3);
        candidate_actions[0] = turn_left(action);
        candidate_actions[1] = action;
        candidate_actions[2] = turn_right(action);
        return candidate_actions[r_util.sample_discrete(slip_action_probs)];
    }

    FourRoomsState FourRooms::apply_action(const FourRoomsState &state, uint32_t action) const {
        int32_t next_x = static_cast<int32_t>(state.x);
        int32_t next_y = static_cast<int32_t>(state.y);

        if (action == LEFT) {
            next_x -= 1;
        } else if (action == DOWN) {
            next_y += 1;
        } else if (action == RIGHT) {
            next_x += 1;
        } else if (action == UP) {
            next_y -= 1;
        } else {
            throw runtime_error("Invalid direction in FourRooms");
        }

        FourRoomsState next_state = state;
        if (!is_inside(next_x, next_y)) {
            return next_state;
        }

        uint32_t bounded_x = static_cast<uint32_t>(next_x);
        uint32_t bounded_y = static_cast<uint32_t>(next_y);
        if (is_wall(bounded_x, bounded_y, state.door_cells)) {
            return next_state;
        }

        next_state.x = bounded_x;
        next_state.y = bounded_y;
        return next_state;
    }

    bool FourRooms::is_goal(const FourRoomsState &state) const {
        return state.x == state.goal_x && state.y == state.goal_y;
    }

    double FourRooms::compute_goal_reward(uint32_t step_count) const {
        return 1. - 0.9 * (static_cast<double>(step_count) / static_cast<double>(FOUR_ROOMS_TIME_LIMIT));
    }

    bool FourRooms::is_inside(int32_t x, int32_t y) const {
        return x >= 0 && y >= 0 &&
               x < static_cast<int32_t>(FOUR_ROOMS_GRID_SIZE) &&
               y < static_cast<int32_t>(FOUR_ROOMS_GRID_SIZE);
    }

    bool FourRooms::is_wall(uint32_t x, uint32_t y,
                            const std::array<uint32_t, FOUR_ROOMS_NUM_DOORS> &door_cells) const {
        if (is_door_cell(encode_cell(x, y), door_cells)) {
            return false;
        }
        return x == FOUR_ROOMS_N || y == FOUR_ROOMS_N;
    }

    bool FourRooms::is_door_cell(uint32_t cell_id,
                                 const std::array<uint32_t, FOUR_ROOMS_NUM_DOORS> &door_cells) const {
        return find(door_cells.begin(), door_cells.end(), cell_id) != door_cells.end();
    }

    bool FourRooms::is_valid_spawn_cell(uint32_t x, uint32_t y,
                                        const std::array<uint32_t, FOUR_ROOMS_NUM_DOORS> &door_cells) const {
        return !is_wall(x, y, door_cells) && !is_door_cell(encode_cell(x, y), door_cells);
    }

    uint32_t FourRooms::encode_cell(uint32_t x, uint32_t y) const {
        return y * FOUR_ROOMS_GRID_SIZE + x;
    }

    std::pair<uint32_t, uint32_t> FourRooms::decode_cell(uint32_t cell_id) const {
        return make_pair(cell_id % FOUR_ROOMS_GRID_SIZE, cell_id / FOUR_ROOMS_GRID_SIZE);
    }

    void FourRooms::validate_state(const FourRoomsState &state) const {
        if (!is_inside(static_cast<int32_t>(state.x), static_cast<int32_t>(state.y))) {
            throw runtime_error("FourRooms agent position is outside the map");
        }
        if (!is_inside(static_cast<int32_t>(state.start_x), static_cast<int32_t>(state.start_y))) {
            throw runtime_error("FourRooms start position is outside the map");
        }
        if (!is_inside(static_cast<int32_t>(state.goal_x), static_cast<int32_t>(state.goal_y))) {
            throw runtime_error("FourRooms goal position is outside the map");
        }
        if (state.time > FOUR_ROOMS_TIME_LIMIT) {
            throw runtime_error("FourRooms state exceeds the time limit");
        }

        for (uint32_t i = 0; i < FOUR_ROOMS_NUM_DOORS; i++) {
            if (!is_valid_door_cell(state.door_cells[i], i)) {
                throw runtime_error("FourRooms door placement is invalid");
            }
        }
        for (uint32_t i = 0; i < FOUR_ROOMS_NUM_DOORS; i++) {
            for (uint32_t j = i + 1; j < FOUR_ROOMS_NUM_DOORS; j++) {
                if (state.door_cells[i] == state.door_cells[j]) {
                    throw runtime_error("FourRooms door placements must be unique");
                }
            }
        }

        if (state.start_x == state.goal_x && state.start_y == state.goal_y) {
            throw runtime_error("FourRooms start and goal must be different");
        }

        if (!is_valid_spawn_cell(state.start_x, state.start_y, state.door_cells)) {
            throw runtime_error("FourRooms start must be placed on a non-wall, non-door cell");
        }
        if (!is_valid_spawn_cell(state.goal_x, state.goal_y, state.door_cells)) {
            throw runtime_error("FourRooms goal must be placed on a non-wall, non-door cell");
        }

        if (is_wall(state.x, state.y, state.door_cells)) {
            throw runtime_error("FourRooms agent cannot occupy a wall cell");
        }
    }

    bool FourRooms::is_valid_door_cell(uint32_t cell_id, uint32_t door_index) const {
        auto position = decode_cell(cell_id);
        uint32_t x = position.first;
        uint32_t y = position.second;

        if (door_index == 0) {
            return x == FOUR_ROOMS_N && y < FOUR_ROOMS_N;
        }
        if (door_index == 1) {
            return x == FOUR_ROOMS_N && y > FOUR_ROOMS_N && y < FOUR_ROOMS_GRID_SIZE;
        }
        if (door_index == 2) {
            return y == FOUR_ROOMS_N && x < FOUR_ROOMS_N;
        }
        if (door_index == 3) {
            return y == FOUR_ROOMS_N && x > FOUR_ROOMS_N && x < FOUR_ROOMS_GRID_SIZE;
        }

        return false;
    }

    const char *FourRooms::format_action(uint32_t action) const {
        static const char *action_names[] = {"LEFT", "DOWN", "RIGHT", "UP"};
        if (action >= num_actions) {
            return "UNKNOWN";
        }
        return action_names[action];
    }

}
