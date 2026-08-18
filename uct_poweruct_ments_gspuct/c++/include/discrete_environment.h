//
// Created by klink on 23.07.19.
//

#ifndef C___DISCRETE_ENVIRONMENT_H
#define C___DISCRETE_ENVIRONMENT_H

#include <cstdint>
#include <vector>
#include <tuple>
#include <random_utils.h>
#include <memory>
#include <util.h>
#include <environment.h>

namespace poweruct {

    class Transition {

    public:
        Transition(std::vector<double> rewards, std::vector<bool> dones, std::vector<uint32_t> next_states,
                   std::vector<double> probabilities, RandomUtils &r_util);

        ~Transition();

        std::tuple<uint32_t, double, bool> sample();

    private:
        std::vector<double> rewards;
        std::vector<bool> dones;
        std::vector<uint32_t> next_states;
        std::vector<double> probabilities;
        RandomUtils &r_util;
    };


    struct DiscreteEnvironmentState {
        DiscreteEnvironmentState() {}

        DiscreteEnvironmentState(uint32_t state, uint32_t time) : state(state), time(time) {}

        uint32_t state;
        uint32_t time;
    };

    class DiscreteEnvironment : public Environment<DiscreteEnvironmentState> {

    public:
        DiscreteEnvironment(uint32_t ns, uint32_t na, double discount_factor, std::vector<Transition> transitions,
                            std::vector<double> initial_states, uint32_t time_limit);

        virtual ~DiscreteEnvironment();

        uint32_t getNumberOfObservations() override;

        uint32_t getNumberOfActions() override;

        void seed(uint32_t s) override;

        void reset() override;

        void set_state(DiscreteEnvironmentState state) override;

        std::tuple<DiscreteEnvironmentState, uint32_t, double, bool> step(uint32_t action) override;

        std::tuple<uint32_t, DiscreteEnvironmentState, uint32_t, double, bool> random_step() override;

        std::tuple<DiscreteEnvironmentState, uint32_t, double, bool>
        simulate(DiscreteEnvironmentState state, uint32_t action) override;

        std::tuple<uint32_t, DiscreteEnvironmentState, uint32_t, double, bool>
        random_simulate(DiscreteEnvironmentState state) override;

        std::unique_ptr<std::vector<std::tuple<uint32_t, double, DiscreteEnvironmentState, uint32_t >>>
        rollout(DiscreteEnvironmentState state) override;

    protected:
        uint32_t state;
        uint32_t last_action;
        uint32_t time_step;

        virtual RandomUtils &get_r_util() = 0;

        virtual std::vector<double>& get_rollout_action_probs(DiscreteEnvironmentState &state);

    private:
        uint32_t ns;
        uint32_t na;
        double discount_factor;
        std::vector<Transition> transitions;
        std::vector<double> initial_states;
        std::vector<double> uniform_action_probs;
        uint32_t time_limit;
        bool initialized;

    };

}

#endif //C___DISCRETE_ENVIRONMENT_H
