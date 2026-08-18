#include <utility>

#include <discrete_environment.h>
#include <iostream>

using namespace std;

namespace poweruct {

    Transition::Transition(vector<double> rewards, vector<bool> dones, vector<uint32_t> next_states,
                           vector<double> probabilities, RandomUtils &r_util) : rewards(move(rewards)),
                                                                                dones(move(dones)),
                                                                                next_states(move(next_states)),
                                                                                probabilities(move(probabilities)),
                                                                                r_util(r_util) {}

    Transition::~Transition() = default;

    tuple<uint32_t, double, bool> Transition::sample() {
        uint32_t id = r_util.sample_discrete(probabilities);
        return make_tuple(next_states[id], rewards[id], dones[id]);
    }

    DiscreteEnvironment::DiscreteEnvironment(uint32_t ns, uint32_t na, double discount_factor,
                                             vector<Transition> transitions, vector<double> initial_states,
                                             uint32_t time_limit) : ns(ns), na(na), discount_factor(discount_factor),
                                                                    transitions(std::move(transitions)),
                                                                    initial_states(move(initial_states)),
                                                                    time_limit(time_limit),
                                                                    uniform_action_probs(na, 1. / ((double) na)) {
        state = 0;
        time_step = 0;
    }

    DiscreteEnvironment::~DiscreteEnvironment() = default;

    uint32_t DiscreteEnvironment::getNumberOfActions() {
        return na;
    }

    uint32_t DiscreteEnvironment::getNumberOfObservations() {
        return ns * time_limit;
    }

    void DiscreteEnvironment::seed(uint32_t s) {
        get_r_util().seed(s);
    }

    void DiscreteEnvironment::reset() {
        state = get_r_util().sample_discrete(initial_states);
        time_step = 0;
        initialized = true;
    }

    void DiscreteEnvironment::set_state(poweruct::DiscreteEnvironmentState state) {
        this->state = state.state;
        this->time_step = state.time;
        initialized = true;
    }

    tuple<DiscreteEnvironmentState, uint32_t, double, bool> DiscreteEnvironment::step(uint32_t action) {
        if (!initialized) {
            throw runtime_error(
                    "Environment has not been initialized! Call reset() or set_state() at least once before calling step()");
        }

        if (time_step >= time_limit) {
            throw runtime_error("Time Limit execeeded!");
        }

        double reward;
        bool done;
        tie(state, reward, done) = transitions[state * na + action].sample();


        time_step += 1;
        last_action = action;

        return make_tuple(DiscreteEnvironmentState(state, time_step), state, reward, done | (time_step == time_limit));
    }

    std::tuple<uint32_t, DiscreteEnvironmentState, uint32_t, double, bool> DiscreteEnvironment::random_step() {
        uint32_t action = get_r_util().sample_discrete(uniform_action_probs);
        auto res = step(action);
        return make_tuple(action, get<0>(res), get<1>(res), get<2>(res), get<3>(res));
    }

    std::tuple<DiscreteEnvironmentState, uint32_t, double, bool> DiscreteEnvironment::simulate(
            DiscreteEnvironmentState state, uint32_t action) {
        set_state(state);
        return step(action);
    }

    std::tuple<uint32_t, DiscreteEnvironmentState, uint32_t, double, bool> DiscreteEnvironment::random_simulate(
            DiscreteEnvironmentState state) {
        set_state(state);
        return random_step();
    }


    std::unique_ptr<std::vector<std::tuple<uint32_t, double, DiscreteEnvironmentState, uint32_t >>>
    DiscreteEnvironment::rollout(DiscreteEnvironmentState state) {
        set_state(state);

        auto res = make_unique<std::vector<std::tuple<uint32_t, double, DiscreteEnvironmentState, uint32_t >>>();
        bool done = false;
        auto cur_state = state;
        while (!done) {
            uint32_t action = get_r_util().sample_discrete(this->get_rollout_action_probs(cur_state));
            uint32_t next_obs;
            double reward;
            tie(cur_state, next_obs, reward, done) = step(action);
            res->emplace_back(make_tuple(action, reward, cur_state, next_obs));
        }

        return res;
    }

    std::vector<double>& DiscreteEnvironment::get_rollout_action_probs(poweruct::DiscreteEnvironmentState &state) {
        return uniform_action_probs;
    }


}