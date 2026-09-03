#include <mc_graph.h>
#include <envs/factored_river_swim.h>
#include <envs/four_rooms.h>
#include <envs/frozen_lake.h>
#include <envs/passenger_grid.h>
#include <envs/sysadmin_ring.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace poweruct;
using namespace std;

namespace {

    static const uint32_t LEFT = 0;
    static const uint32_t DOWN = 1;
    static const uint32_t RIGHT = 2;
    static const uint32_t UP = 3;

    static const char *grid_action_names[] = {"LEFT", "DOWN", "RIGHT", "UP"};

    static const char *frozen_lake_map = "SFFFFFFF"
                                         "FFFFFFFF"
                                         "FFFHFFFF"
                                         "FFFFFHFF"
                                         "FFFHFFFF"
                                         "FHHFFFHF"
                                         "FHFFHFHF"
                                         "FFFHFFFG";

    uint32_t turn_left(uint32_t action) {
        return (action + 3) % 4;
    }

    uint32_t turn_right(uint32_t action) {
        return (action + 1) % 4;
    }

    string factored_river_swim_action_name(uint32_t action) {
        if (action >= (1U << FACTORED_RIVER_SWIM_NUM_RIVERS)) {
            return string("UNKNOWN(") + to_string(action) + ")";
        }

        ostringstream out;
        out << "A[";
        for (uint32_t river = 0; river < FACTORED_RIVER_SWIM_NUM_RIVERS; river++) {
            if (river > 0) {
                out << ", ";
            }
            out << ((action >> river) & 1U);
        }
        out << "]";
        return out.str();
    }

    string sysadmin_ring_action_name(uint32_t action) {
        if (action < SYSADMIN_RING_NUM_COMPUTERS) {
            return string("REBOOT[") + to_string(action) + "]";
        }
        if (action == SYSADMIN_RING_NUM_COMPUTERS) {
            return "IDLE";
        }
        return string("UNKNOWN(") + to_string(action) + ")";
    }

    string action_name(uint32_t action, const string &environment_name) {
        if (environment_name == "factored_river_swim") {
            return factored_river_swim_action_name(action);
        }

        if (environment_name == "sysadmin_ring") {
            return sysadmin_ring_action_name(action);
        }

        if (action >= 4) {
            return string("UNKNOWN(") + to_string(action) + ")";
        }
        return grid_action_names[action];
    }

    string canonical_env_name(string env_name) {
        if (env_name == "frozenlake") {
            return "frozen_lake";
        }
        if (env_name == "passengergrid") {
            return "passenger_grid";
        }
        if (env_name == "factoredriverswim") {
            return "factored_river_swim";
        }
        if (env_name == "fourrooms") {
            return "four_rooms";
        }
        if (env_name == "sysadminring") {
            return "sysadmin_ring";
        }
        return env_name;
    }

    GraphSearchMode parse_graph_mode(const string &mode_name) {
        if (mode_name == "gs_power_uct" || mode_name == "gs_power_uct_er") {
            return GraphSearchMode::Depth;
        }
        if (mode_name == "gs_power_uct_f") {
            return GraphSearchMode::Full;
        }

        throw runtime_error("Mode must be gs_power_uct, gs_power_uct_f or gs_power_uct_er");
    }

    // No '_er_f': the recursive resistance needs the layered depth-augmented graph.
    bool is_er_algorithm(const string &mode_name) {
        return mode_name == "gs_power_uct_er";
    }

    struct GraphRunArgs {
        uint32_t n_rollouts = 0;
        double c = 0.;
        double p = 0.;
        double c2 = 0.;
        double c3 = 0.;
        uint32_t seed = 0;
    };

    uint32_t parse_uint32(const char *raw, const string &argument_name) {
        size_t processed = 0;
        unsigned long parsed = stoul(raw, &processed);
        if (raw[processed] != '\0') {
            throw runtime_error("Invalid integer for " + argument_name + ": " + string(raw));
        }
        if (parsed > numeric_limits<uint32_t>::max()) {
            throw runtime_error("Integer out of range for " + argument_name + ": " + string(raw));
        }
        return static_cast<uint32_t>(parsed);
    }

    double parse_double(const char *raw, const string &argument_name) {
        size_t processed = 0;
        double parsed = stod(raw, &processed);
        if (raw[processed] != '\0') {
            throw runtime_error("Invalid floating point value for " + argument_name + ": " + string(raw));
        }
        return parsed;
    }

    /**
     * Layout: ALGO ENV [TIME_LIMIT] N_ROLLOUTS C P [C2 C3] [SEED], where the ER coefficients are
     * present exactly for the '_er' algorithms, so the arity is unambiguous.
     */
    GraphRunArgs parse_graph_run_args(int argc, char *argv[], int first_index, bool er) {
        int expected = first_index + 3 + (er ? 2 : 0);
        if (argc != expected && argc != expected + 1) {
            throw runtime_error("Wrong number of arguments for this algorithm/environment pair");
        }

        GraphRunArgs args;
        args.n_rollouts = parse_uint32(argv[first_index], "N_ROLLOUTS");
        args.c = parse_double(argv[first_index + 1], "C");
        args.p = parse_double(argv[first_index + 2], "P");
        if (er) {
            args.c2 = parse_double(argv[first_index + 3], "C2");
            args.c3 = parse_double(argv[first_index + 4], "C3");
        }
        args.seed = argc == expected + 1 ? parse_uint32(argv[argc - 1], "SEED") : 0;
        return args;
    }

    uint32_t apply_frozen_lake_action(uint32_t state_id, uint32_t action) {
        uint32_t x = state_id % 8;
        uint32_t y = state_id / 8;

        if (action == LEFT) {
            if (x > 0) {
                x -= 1;
            }
        } else if (action == DOWN) {
            if (y < 7) {
                y += 1;
            }
        } else if (action == RIGHT) {
            if (x < 7) {
                x += 1;
            }
        } else if (action == UP) {
            if (y > 0) {
                y -= 1;
            }
        } else {
            throw runtime_error("Invalid FrozenLake action");
        }

        return 8 * y + x;
    }

    PassengerGridState apply_passenger_grid_action(const PassengerGridState &state, uint32_t action) {
        PassengerGridState next_state = state;

        if (action == LEFT) {
            if (next_state.x > 0) {
                next_state.x -= 1;
            }
        } else if (action == DOWN) {
            if (next_state.y + 1 < 6) {
                next_state.y += 1;
            }
        } else if (action == RIGHT) {
            if (next_state.x + 1 < 7) {
                next_state.x += 1;
            }
        } else if (action == UP) {
            if (next_state.y > 0) {
                next_state.y -= 1;
            }
        } else {
            throw runtime_error("Invalid PassengerGrid action");
        }

        return next_state;
    }

    bool is_four_rooms_door(uint32_t cell_id, const array<uint32_t, FOUR_ROOMS_NUM_DOORS> &door_cells) {
        return find(door_cells.begin(), door_cells.end(), cell_id) != door_cells.end();
    }

    bool is_four_rooms_wall(uint32_t x, uint32_t y, const array<uint32_t, FOUR_ROOMS_NUM_DOORS> &door_cells) {
        uint32_t cell_id = y * FOUR_ROOMS_GRID_SIZE + x;
        if (is_four_rooms_door(cell_id, door_cells)) {
            return false;
        }
        return x == FOUR_ROOMS_N || y == FOUR_ROOMS_N;
    }

    FourRoomsState apply_four_rooms_action(const FourRoomsState &state, uint32_t action) {
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
            throw runtime_error("Invalid FourRooms action");
        }

        FourRoomsState next_state = state;
        next_state.time += 1;
        if (next_x < 0 || next_y < 0 ||
            next_x >= static_cast<int32_t>(FOUR_ROOMS_GRID_SIZE) ||
            next_y >= static_cast<int32_t>(FOUR_ROOMS_GRID_SIZE)) {
            return next_state;
        }

        if (is_four_rooms_wall(static_cast<uint32_t>(next_x), static_cast<uint32_t>(next_y), state.door_cells)) {
            return next_state;
        }

        next_state.x = static_cast<uint32_t>(next_x);
        next_state.y = static_cast<uint32_t>(next_y);
        return next_state;
    }

    vector<uint32_t> deduplicate_actions(vector<uint32_t> actions) {
        sort(actions.begin(), actions.end());
        actions.erase(unique(actions.begin(), actions.end()), actions.end());
        return actions;
    }

    vector<uint32_t> infer_effective_actions(const DiscreteEnvironmentState &state, uint32_t intended_action,
                                             const DiscreteEnvironmentState &next_state) {
        vector<uint32_t> candidates;
        vector<uint32_t> slipped_actions({turn_left(intended_action), intended_action, turn_right(intended_action)});
        for (uint32_t action : slipped_actions) {
            if (apply_frozen_lake_action(state.state, action) == next_state.state) {
                candidates.emplace_back(action);
            }
        }
        return deduplicate_actions(candidates);
    }

    vector<uint32_t> infer_effective_actions(const PassengerGridState &state, uint32_t intended_action,
                                             const PassengerGridState &next_state) {
        vector<uint32_t> candidates;
        vector<uint32_t> slipped_actions({turn_left(intended_action), intended_action, turn_right(intended_action)});
        for (uint32_t action : slipped_actions) {
            PassengerGridState candidate_state = apply_passenger_grid_action(state, action);
            if (candidate_state.x == next_state.x && candidate_state.y == next_state.y) {
                candidates.emplace_back(action);
            }
        }
        return deduplicate_actions(candidates);
    }

    vector<uint32_t> infer_effective_actions(const FourRoomsState &state, uint32_t intended_action,
                                             const FourRoomsState &next_state) {
        vector<uint32_t> candidates;
        vector<uint32_t> slipped_actions({turn_left(intended_action), intended_action, turn_right(intended_action)});
        for (uint32_t action : slipped_actions) {
            FourRoomsState candidate_state = apply_four_rooms_action(state, action);
            if (candidate_state.x == next_state.x && candidate_state.y == next_state.y) {
                candidates.emplace_back(action);
            }
        }
        return deduplicate_actions(candidates);
    }

    vector<uint32_t> infer_effective_actions(const FactoredRiverSwimState &, uint32_t intended_action,
                                             const FactoredRiverSwimState &) {
        return vector<uint32_t>({intended_action});
    }

    vector<uint32_t> infer_effective_actions(const SysAdminRingState &, uint32_t intended_action,
                                             const SysAdminRingState &) {
        return vector<uint32_t>({intended_action});
    }

    string format_action_candidates(const vector<uint32_t> &actions, const string &environment_name) {
        if (actions.empty()) {
            return "effective_action=UNKNOWN";
        }

        if (actions.size() == 1) {
            return string("effective_action=") + action_name(actions.front(), environment_name);
        }

        ostringstream out;
        out << "effective_action_candidates={";
        for (size_t i = 0; i < actions.size(); i++) {
            if (i > 0) {
                out << ", ";
            }
            out << action_name(actions[i], environment_name);
        }
        out << "}";
        return out.str();
    }

    string format_passenger_mask(uint32_t passenger_mask) {
        ostringstream out;
        out << "0b";
        for (int bit = 2; bit >= 0; bit--) {
            out << (((passenger_mask >> bit) & 1U) != 0U ? '1' : '0');
        }
        return out.str();
    }

    string describe_state(const DiscreteEnvironmentState &state) {
        ostringstream out;
        out << "{id=" << state.state
            << ", x=" << (state.state % 8)
            << ", y=" << (state.state / 8)
            << ", tile=" << frozen_lake_map[state.state]
            << ", time=" << state.time
            << "}";
        return out.str();
    }

    string describe_state(const PassengerGridState &state) {
        ostringstream out;
        out << "{x=" << state.x
            << ", y=" << state.y
            << ", passenger_mask=" << format_passenger_mask(state.passenger_mask)
            << ", time=" << state.time
            << "}";
        return out.str();
    }

    string describe_state(const FourRoomsState &state) {
        ostringstream out;
        out << "{x=" << state.x
            << ", y=" << state.y
            << ", rel_x=" << (static_cast<int32_t>(state.x) - static_cast<int32_t>(state.start_x))
            << ", rel_y=" << (static_cast<int32_t>(state.y) - static_cast<int32_t>(state.start_y))
            << ", start=(" << state.start_x << ", " << state.start_y << ")"
            << ", goal=(" << state.goal_x << ", " << state.goal_y << ")"
            << ", time=" << state.time
            << "}";
        return out.str();
    }

    string describe_state(const FactoredRiverSwimState &state) {
        ostringstream out;
        out << "{positions=[";
        for (uint32_t river = 0; river < FACTORED_RIVER_SWIM_NUM_RIVERS; river++) {
            if (river > 0) {
                out << ", ";
            }
            out << state.positions[river];
        }
        out << "], time=" << state.time << "}";
        return out.str();
    }

    string describe_state(const SysAdminRingState &state) {
        ostringstream out;
        out << "{alive_mask=0b";
        for (int machine = static_cast<int>(SYSADMIN_RING_NUM_COMPUTERS) - 1; machine >= 0; machine--) {
            out << (((state.alive_mask >> machine) & 1U) != 0U ? '1' : '0');
        }
        out << ", time=" << state.time << "}";
        return out.str();
    }

    void print_usage(const char *program_name) {
        cout << "Usage:\n";
        cout << "  " << program_name << " ALGO ENV N_ROLLOUTS C P [SEED]\n";
        cout << "  " << program_name << " gs_power_uct_er ENV N_ROLLOUTS C P C2 C3 [SEED]\n";
        cout << "  " << program_name << " ALGO passenger_grid TIME_LIMIT N_ROLLOUTS C P [SEED]\n";
        cout << "  " << program_name << " gs_power_uct_er passenger_grid TIME_LIMIT N_ROLLOUTS C P C2 C3 [SEED]\n";
        cout << "ALGO: gs_power_uct, gs_power_uct_f, gs_power_uct_er (depth-augmented graph only)\n";
        cout << "ENV: frozen_lake, factored_river_swim, four_rooms, sysadmin_ring, passenger_grid\n";
        cout << "Accepted environment aliases: frozenlake, factoredriverswim, fourrooms, passengergrid, sysadminring\n";
    }

    template<typename S>
    shared_ptr<MCGraphSearchTree<S>> create_mc_graph_tree(shared_ptr<Environment<S>> env_search, GraphSearchMode mode,
                                                          const GraphRunArgs &args) {
        return make_shared<MCGraphSearchTree<S>>(env_search, env_search->getInitialState(),
                                                 env_search->getInitialObservation(), 1., args.c, args.p, mode,
                                                 args.c2, args.c3);
    }

    string format_action_stats(const vector<GraphActionStats> &action_stats, const string &environment_name) {
        ostringstream out;
        out << "{";
        for (size_t i = 0; i < action_stats.size(); i++) {
            if (i > 0) {
                out << ", ";
            }

            out << action_name(action_stats[i].action, environment_name)
                << ":(Q=" << action_stats[i].q_value
                << ", visits=" << action_stats[i].visits
                << ")";
        }
        out << "}";
        return out.str();
    }

    template<typename S>
    double run_single_episode(const string &mode_name, const string &environment_name, const GraphRunArgs &args,
                              shared_ptr<Environment<S>> env, shared_ptr<Environment<S>> env_search,
                              GraphSearchMode mode, bool plan_once, bool reset_before_tree) {
        uint32_t n_rollouts = args.n_rollouts;
        uint32_t seed = args.seed;

        env->seed(seed);
        env_search->seed(seed);
        if (reset_before_tree) {
            env_search->reset();
            env->reset();
        }

        auto search_tree = create_mc_graph_tree<S>(env_search, mode, args);
        search_tree->seed(seed);

        if (plan_once) {
            search_tree->search(n_rollouts);
        }

        if (!reset_before_tree) {
            env->reset();
        }
        S current_state = env->getInitialState();

        cout << "mode=" << mode_name
             << ", environment=" << environment_name
             << ", search_budget=" << n_rollouts
             << ", planning_mode=" << (plan_once ? "plan_once" : "replan")
             << ", c=" << args.c
             << ", p=" << args.p
             << ", c2=" << args.c2
             << ", c3=" << args.c3
             << ", seed=" << seed
             << endl;
        cout << "initial_state=" << describe_state(current_state) << endl;

        bool done = false;
        double total_reward = 0.;
        uint32_t step_index = 0;
        while (!done) {
            uint32_t action = search_tree->search(plan_once ? 0 : n_rollouts);
            auto action_stats = search_tree->get_current_action_stats();

            cout << "step=" << step_index
                 << ", q_values_before_action_selection=" << format_action_stats(action_stats, environment_name)
                 << endl;

            S next_state;
            uint32_t next_observation;
            double reward;
            tie(next_state, next_observation, reward, done) = env->step(action);

            auto effective_actions = infer_effective_actions(current_state, action, next_state);
            cout << "step=" << step_index
                 << ", state=" << describe_state(current_state)
                 << ", chosen_action=" << action_name(action, environment_name)
                 << ", " << format_action_candidates(effective_actions, environment_name)
                 << ", next_state=" << describe_state(next_state)
                 << ", reward=" << reward
                 << ", done=" << (done ? "true" : "false")
                 << endl;

            total_reward += reward;
            search_tree->progress_tree(action, next_state, next_observation);
            current_state = next_state;
            step_index += 1;
        }

        cout << "episode_return=" << total_reward << endl;
        cout << "steps_taken=" << step_index << endl;

        return total_reward;
    }

}

int main(int argc, char *argv[]) {
    try {
        if (argc < 6) {
            print_usage(argv[0]);
            return -1;
        }

        string mode_name(argv[1]);
        string environment_name = canonical_env_name(argv[2]);
        auto mode = parse_graph_mode(mode_name);
        bool er = is_er_algorithm(mode_name);

        if (environment_name == "frozen_lake") {
            auto args = parse_graph_run_args(argc, argv, 3, er);

            auto env = make_shared<FrozenLake>(true, false);
            auto env_search = make_shared<FrozenLake>(true, false);
            run_single_episode<DiscreteEnvironmentState>(mode_name, environment_name, args,
                                                         static_pointer_cast<Environment<DiscreteEnvironmentState>>(env),
                                                         static_pointer_cast<Environment<DiscreteEnvironmentState>>(env_search),
                                                         mode, false, false);
            return 0;
        }

        if (environment_name == "factored_river_swim") {
            auto args = parse_graph_run_args(argc, argv, 3, er);

            auto env = make_shared<FactoredRiverSwim>();
            auto env_search = make_shared<FactoredRiverSwim>();
            run_single_episode<FactoredRiverSwimState>(mode_name, environment_name, args,
                                                       static_pointer_cast<Environment<FactoredRiverSwimState>>(env),
                                                       static_pointer_cast<Environment<FactoredRiverSwimState>>(env_search),
                                                       mode, false, false);
            return 0;
        }

        if (environment_name == "four_rooms") {
            auto args = parse_graph_run_args(argc, argv, 3, er);

            auto env = make_shared<FourRooms>();
            auto env_search = make_shared<FourRooms>();
            run_single_episode<FourRoomsState>(mode_name, environment_name, args,
                                               static_pointer_cast<Environment<FourRoomsState>>(env),
                                               static_pointer_cast<Environment<FourRoomsState>>(env_search),
                                               mode, false, true);
            return 0;
        }

        if (environment_name == "sysadmin_ring") {
            auto args = parse_graph_run_args(argc, argv, 3, er);

            auto env = make_shared<SysAdminRing>();
            auto env_search = make_shared<SysAdminRing>();
            run_single_episode<SysAdminRingState>(mode_name, environment_name, args,
                                                  static_pointer_cast<Environment<SysAdminRingState>>(env),
                                                  static_pointer_cast<Environment<SysAdminRingState>>(env_search),
                                                  mode, false, false);
            return 0;
        }

        if (environment_name == "passenger_grid") {
            uint32_t time_limit = parse_uint32(argv[3], "TIME_LIMIT");
            auto args = parse_graph_run_args(argc, argv, 4, er);

            auto env = make_shared<PassengerGrid>(time_limit, true);
            auto env_search = make_shared<PassengerGrid>(time_limit, true);
            run_single_episode<PassengerGridState>(mode_name, environment_name, args,
                                                   static_pointer_cast<Environment<PassengerGridState>>(env),
                                                   static_pointer_cast<Environment<PassengerGridState>>(env_search),
                                                   mode, false, false);
            return 0;
        }

        throw runtime_error(
                "Environment must be frozen_lake, factored_river_swim, four_rooms, sysadmin_ring or passenger_grid");
    } catch (const exception &ex) {
        cerr << "testmcgraph failed: " << ex.what() << endl;
        print_usage(argv[0]);
        return -1;
    }
}
