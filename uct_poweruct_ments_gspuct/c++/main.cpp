#include <iostream>
#include <tree_search.h>
#include <mc_graph.h>
#include <envs/frozen_lake.h>
#include <envs/factored_river_swim.h>
#include <envs/four_rooms.h>
#include <envs/passenger_grid.h>
#include <envs/sysadmin_ring.h>
#include <mpi.h>
#include <functional>
#include <memory>
#include <map>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace poweruct;
using namespace std;

struct PerformanceStats {
    double mean = 0.;
    double std = 0.;
};

PerformanceStats compute_performance_stats(const double *rewards, uint32_t n_experiments) {
    PerformanceStats stats;
    if (n_experiments == 0) {
        return stats;
    }

    for (uint32_t i = 0; i < n_experiments; i++) {
        stats.mean += rewards[i];
    }
    stats.mean /= static_cast<double>(n_experiments);

    if (n_experiments == 1) {
        return stats;
    }

    double squared_deviation_sum = 0.;
    for (uint32_t i = 0; i < n_experiments; i++) {
        double deviation = rewards[i] - stats.mean;
        squared_deviation_sum += deviation * deviation;
    }

    stats.std = sqrt(squared_deviation_sum / static_cast<double>(n_experiments - 1));
    return stats;
}

uint32_t get_algorithm_required_argc(const string &type) {
    if (type == "max_uct") {
        return 5;
    }
    if (type == "power_uct" || type == "ments" ||
        type == "gs_power_uct" || type == "gs_power_uct_f") {
        return 6;
    }

    throw runtime_error(string("Invalid algorithm type: ") + type);
}

uint32_t get_env_extra_argc(const string &env_name) {
    if (env_name == "frozen_lake" || env_name == "factored_river_swim" ||
        env_name == "four_rooms" || env_name == "sysadmin_ring") {
        return 0;
    }
    if (env_name == "passenger_grid") {
        return 1;
    }

    throw runtime_error(string("Invalid environment: ") + env_name);
}

uint32_t get_required_argc(const string &type, const string &env_name) {
    return get_algorithm_required_argc(type) + get_env_extra_argc(env_name);
}

vector<string> get_algorithm_parameter_names(const string &type) {
    if (type == "max_uct") {
        return {"alpha"};
    }
    if (type == "power_uct") {
        return {"alpha", "p"};
    }
    if (type == "gs_power_uct" || type == "gs_power_uct_f") {
        return {"c", "p"};
    }
    if (type == "ments") {
        return {"tau", "epsilon"};
    }

    throw runtime_error(string("Invalid algorithm type: ") + type);
}

vector<pair<string, string>> get_run_parameters(int argc, char *argv[]) {
    vector<pair<string, string>> params;
    if (argc <= 1) {
        return params;
    }

    params.emplace_back("n_experiments", argv[1]);
    if (argc <= 2) {
        return params;
    }

    params.emplace_back("algorithm", argv[2]);
    if (argc <= 3) {
        return params;
    }

    params.emplace_back("environment", argv[3]);

    uint32_t next_arg_index = 4;
    string env_name(argv[3]);
    if (env_name == "factored_river_swim") {
        params.emplace_back("num_rivers", to_string(FACTORED_RIVER_SWIM_NUM_RIVERS));
        params.emplace_back("nlocations", to_string(FACTORED_RIVER_SWIM_NUM_LOCATIONS));
        params.emplace_back("time_limit", to_string(FACTORED_RIVER_SWIM_TIME_LIMIT));
    } else if (env_name == "four_rooms") {
        params.emplace_back("n", to_string(FOUR_ROOMS_N));
        params.emplace_back("grid_size", to_string(FOUR_ROOMS_GRID_SIZE));
        params.emplace_back("time_limit", to_string(FOUR_ROOMS_TIME_LIMIT));
    } else if (env_name == "sysadmin_ring") {
        params.emplace_back("ncomps", to_string(SYSADMIN_RING_NUM_COMPUTERS));
        params.emplace_back("time_limit", to_string(SYSADMIN_RING_TIME_LIMIT));
    } else if (env_name == "passenger_grid" && argc > 4) {
        params.emplace_back("time_limit", argv[4]);
        next_arg_index = 5;
    }

    auto parameter_names = get_algorithm_parameter_names(string(argv[2]));
    for (uint32_t i = 0; i < parameter_names.size() && next_arg_index + i < static_cast<uint32_t>(argc); i++) {
        params.emplace_back(parameter_names[i], argv[next_arg_index + i]);
    }

    return params;
}

template<typename S>
shared_ptr<AbstractSearchTree<S>> create_classic_search_tree(shared_ptr<Environment<S>> search_env,
                                                             double discount_factor, string type, char **argv) {
    if (type == "max_uct") {
        auto alpha = stod(argv[0]);
        return make_shared<MaxUCTSearchTree<S>>(search_env, search_env->getInitialState(),
                                                search_env->getInitialObservation(), discount_factor, alpha);
    }
    if (type == "power_uct") {
        auto alpha = stod(argv[0]);
        auto p = stod(argv[1]);
        return make_shared<PowerUCTSearchTree<S>>(search_env, search_env->getInitialState(),
                                                  search_env->getInitialObservation(), discount_factor, alpha, p);
    }
    if (type == "ments") {
        auto tau = stod(argv[0]);
        auto epsilon = stod(argv[1]);
        return make_shared<MaxEntropySearchTree<S>>(search_env, search_env->getInitialState(),
                                                    search_env->getInitialObservation(), discount_factor, tau, epsilon);
    }

    throw runtime_error(string("Invalid algorithm type: ") + type);
}

template<typename S>
shared_ptr<AbstractSearchTree<S>> create_search_tree(shared_ptr<Environment<S>> search_env, double discount_factor,
                                                     string type, char **argv) {
    if (type == "gs_power_uct") {
        auto c = stod(argv[0]);
        auto p = stod(argv[1]);
        return make_shared<MCGraphSearchTree<S>>(search_env, search_env->getInitialState(),
                                                search_env->getInitialObservation(),
                                                discount_factor, c, p, GraphSearchMode::Depth);
    }
    if (type == "gs_power_uct_f") {
        auto c = stod(argv[0]);
        auto p = stod(argv[1]);
        return make_shared<MCGraphSearchTree<S>>(search_env, search_env->getInitialState(),
                                                search_env->getInitialObservation(),
                                                discount_factor, c, p, GraphSearchMode::Full);
    }

    return create_classic_search_tree<S>(search_env, discount_factor, type, argv);
}

template<typename S>
double run_experiment(uint32_t n_search, uint32_t seed, shared_ptr<Environment<S>> env,
                      shared_ptr<Environment<S>> env_search, double discount_factor, string st_type,
                      char *raw_st_args[], bool replan) {
    auto st = create_search_tree<S>(env_search, discount_factor, st_type, raw_st_args);

    bool done = false;
    double reward = 0.;
    double disc_acc = 1.;
    env_search->seed(seed);
    env->seed(seed);
    st->seed(seed);

    if (!replan) {
        st->search(n_search);
    }

    env->reset();
    while (!done) {
        uint32_t action = st->search(replan ? n_search : 0);
        S next_state;
        uint32_t next_obs;
        double cur_reward;
        tie(next_state, next_obs, cur_reward, done) = env->step(action);
        reward += disc_acc * cur_reward;
        disc_acc *= discount_factor;
        st->progress_tree(action, next_state, next_obs);
    }

    return reward;
}

template<typename S>
double run_randomized_initial_experiment(uint32_t n_search, uint32_t seed, shared_ptr<Environment<S>> env,
                                         shared_ptr<Environment<S>> env_search, double discount_factor,
                                         string st_type, char *raw_st_args[], bool replan) {
    bool done = false;
    double reward = 0.;
    double disc_acc = 1.;
    env_search->seed(seed);
    env->seed(seed);
    env_search->reset();
    env->reset();

    auto st = create_search_tree<S>(env_search, discount_factor, st_type, raw_st_args);
    st->seed(seed);

    if (!replan) {
        st->search(n_search);
    }

    while (!done) {
        uint32_t action = st->search(replan ? n_search : 0);
        S next_state;
        uint32_t next_obs;
        double cur_reward;
        tie(next_state, next_obs, cur_reward, done) = env->step(action);
        reward += disc_acc * cur_reward;
        disc_acc *= discount_factor;
        st->progress_tree(action, next_state, next_obs);
    }

    return reward;
}

std::function<double(uint32_t, uint32_t)> create_experiment_runner(char **argv) {
    return [argv](uint32_t n_search, uint32_t seed) {
        string st_type(argv[1]);
        string env_name(argv[2]);
        char **raw_st_args = argv + 3;

        if (env_name == "frozen_lake") {
            auto env = make_shared<FrozenLake>(true, false);
            auto env_search = make_shared<FrozenLake>(true, false);
            return run_experiment(n_search, seed, static_pointer_cast<Environment<DiscreteEnvironmentState>>(env),
                                  static_pointer_cast<Environment<DiscreteEnvironmentState>>(env_search), 1., st_type,
                                  raw_st_args, true);
        }
        if (env_name == "factored_river_swim") {
            auto env = make_shared<FactoredRiverSwim>();
            auto env_search = make_shared<FactoredRiverSwim>();
            return run_experiment(n_search, seed, static_pointer_cast<Environment<FactoredRiverSwimState>>(env),
                                  static_pointer_cast<Environment<FactoredRiverSwimState>>(env_search), 1., st_type,
                                  raw_st_args, true);
        }
        if (env_name == "four_rooms") {
            auto env = make_shared<FourRooms>();
            auto env_search = make_shared<FourRooms>();
            return run_randomized_initial_experiment(n_search, seed,
                                                     static_pointer_cast<Environment<FourRoomsState>>(env),
                                                     static_pointer_cast<Environment<FourRoomsState>>(env_search),
                                                     1., st_type, raw_st_args, true);
        }
        if (env_name == "sysadmin_ring") {
            auto env = make_shared<SysAdminRing>();
            auto env_search = make_shared<SysAdminRing>();
            return run_experiment(n_search, seed, static_pointer_cast<Environment<SysAdminRingState>>(env),
                                  static_pointer_cast<Environment<SysAdminRingState>>(env_search), 1., st_type,
                                  raw_st_args, true);
        }
        if (env_name == "passenger_grid") {
            auto time_limit = static_cast<uint32_t>(stoul(argv[3]));
            raw_st_args = argv + 4;

            auto env = make_shared<PassengerGrid>(time_limit, true);
            auto env_search = make_shared<PassengerGrid>(time_limit, true);
            return run_experiment(n_search, seed, static_pointer_cast<Environment<PassengerGridState>>(env),
                                  static_pointer_cast<Environment<PassengerGridState>>(env_search), 1., st_type,
                                  raw_st_args, true);
        }

        throw runtime_error(string("Invalid environment: ") + env_name);
    };
}

int main(int argc, char *argv[]) {
    bool error = false;
    if (argc < 5) {
        error = true;
    } else {
        string type(argv[2]);
        string env(argv[3]);
        try {
            if (argc < static_cast<int>(get_required_argc(type, env))) {
                error = true;
            }
        } catch (const runtime_error &) {
            error = true;
        }
    }

    if (error) {
        cout << "Missing required arguments - need one of the following: " << endl;
        cout << "\t\tN_EXPERIMENTS max_uct ENV ALPHA" << endl;
        cout << "\t\tN_EXPERIMENTS power_uct ENV ALPHA P" << endl;
        cout << "\t\tN_EXPERIMENTS gs_power_uct ENV C P" << endl;
        cout << "\t\tN_EXPERIMENTS gs_power_uct_f ENV C P" << endl;
        cout << "\t\tN_EXPERIMENTS ments ENV TAU EPSILON" << endl;
        cout << "\t\tENV=factored_river_swim uses fixed compile-time parameters with no extra env arguments" << endl;
        cout << "\t\tENV=four_rooms uses fixed compile-time parameters with no extra env arguments" << endl;
        cout << "\t\tENV=sysadmin_ring uses fixed compile-time parameters with no extra env arguments" << endl;
        cout << "\t\tFor ENV=passenger_grid add TIME_LIMIT immediately after ENV" << endl;
        return -1;
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto experiment_runner = create_experiment_runner(argv + 1);

    string env(argv[3]);
    map<string, vector<uint32_t>> rollout_map;
    // rollout_map.insert(make_pair("frozen_lake", vector<uint32_t>({512, 1024, 2048, 4096, 8192, 16384})));
    // rollout_map.insert(make_pair("factored_river_swim", vector<uint32_t>({16, 32, 64, 128, 256, 512, 1024,
    //                                                                       2048})));
    // rollout_map.insert(make_pair("four_rooms", vector<uint32_t>({64, 128, 256, 512, 1024, 2048, 4096})));
    // rollout_map.insert(make_pair("sysadmin_ring", vector<uint32_t>({512, 1024, 2048, 4096, 8192, 16384})));
    // rollout_map.insert(make_pair("passenger_grid", vector<uint32_t>({256, 512, 1024, 2048, 4096, 8192})));
    rollout_map.insert(make_pair("frozen_lake", vector<uint32_t>({16, 32, 64, 128, 256, 512, 1024,2048})));
    rollout_map.insert(make_pair("factored_river_swim", vector<uint32_t>({16, 32, 64, 128, 256, 512, 1024,2048
                                                                        })));
    rollout_map.insert(make_pair("four_rooms", vector<uint32_t>({16, 32, 64, 128, 256, 512, 1024,2048})));
    rollout_map.insert(make_pair("sysadmin_ring", vector<uint32_t>({16, 32, 64, 128, 256, 512, 1024,2048})));
    rollout_map.insert(make_pair("passenger_grid", vector<uint32_t>({16, 32, 64, 128, 256, 512, 1024,2048})));

    uint32_t n_experiments = (uint32_t) stoi(argv[1]);
    uint32_t sub_size = size > 1 ? n_experiments / (size - 1) : n_experiments;
    uint32_t remainder = size > 1 ? n_experiments % (size - 1) : 0;

    if (rank == 0) {
        auto rewards = new double[n_experiments];
        auto run_parameters = get_run_parameters(argc, argv);
        vector<pair<uint32_t, PerformanceStats>> performance_summary;
        auto rollouts_ptr = rollout_map.find(env);
        if (rollouts_ptr == rollout_map.end()) {
            cout << "Invalid environment: " << env << endl;
            return -1;
        }
        auto rollouts = rollouts_ptr->second;
        for (auto n_rollouts : rollouts) {
            if (size == 1) {
                for (uint32_t i = 0; i < n_experiments; i++) {
                    rewards[i] = experiment_runner(n_rollouts, i);
                }
            } else {
                for (uint32_t i = 1; i < size; i++) {
                    MPI_Send(&n_rollouts, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
                }

                uint32_t offset = 0;
                for (uint32_t i = 1; i < size; i++) {
                    uint32_t n_sub_experiments = sub_size + (i <= remainder ? 1 : 0);
                    uint32_t buffer_size = n_sub_experiments;
                    double buffer[buffer_size];
                    MPI_Recv(buffer, buffer_size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, NULL);

                    for (uint32_t j = 0; j < n_sub_experiments; j++) {
                        rewards[offset + j] = buffer[j];
                    }
                    offset += n_sub_experiments;
                }
            }

            auto stats = compute_performance_stats(rewards, n_experiments);
            double two_std = 2. * stats.std;
            cout << "Performance (" << n_rollouts << "): mean_reward=" << stats.mean
                 << ", 2std=" << two_std
                 << ", mean+-2std=[" << (stats.mean - two_std) << ", " << (stats.mean + two_std) << "]" << endl;
            performance_summary.emplace_back(n_rollouts, stats);
        }

        cout << "\n===== Run Summary =====" << endl;
        for (const auto &parameter : run_parameters) {
            cout << parameter.first << ": " << parameter.second << endl;
        }
        cout << "mean_performance_by_simulations:" << endl;
        for (const auto &entry : performance_summary) {
            double two_std = 2. * entry.second.std;
            cout << "simulations=" << entry.first
                 << ", mean_reward=" << entry.second.mean
                 << ", std_reward=" << entry.second.std
                 << ", 2std=" << two_std << endl;
        }
        cout << "=======================" << endl;

        delete[] rewards;
        for (uint32_t i = 1; i < size; i++) {
            MPI_Send(NULL, 0, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
        }

    } else {
        uint32_t n_sub_experiments = sub_size + (rank <= remainder ? 1 : 0);
        uint32_t seed_offset = n_experiments * rank;
        uint32_t buffer_size = n_sub_experiments;
        auto *rewards = new double[buffer_size];

        while (true) {
            MPI_Status status;
            MPI_Probe(0, 0, MPI_COMM_WORLD, &status);

            int count = 0;
            MPI_Get_count(&status, MPI_UNSIGNED, &count);
            if (count == 0) {
                break;
            }

            uint32_t n_rollouts;
            MPI_Recv(&n_rollouts, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, NULL);

            cout << "Received command to execute experiments with " << n_rollouts << " simulations" << endl;

            for (uint32_t i = 0; i < n_sub_experiments; i++) {
                rewards[i] = experiment_runner(n_rollouts, seed_offset + i);
            }

            cout << "Rank " << rank << " sending " << n_sub_experiments << " results" << endl;
            MPI_Send(rewards, buffer_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }

        cout << "Shutting down the worker" << endl;
        delete[] rewards;
    }

    MPI_Finalize();
    return 0;
}
