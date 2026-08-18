#ifndef C_ENVIRONMENT_H
#define C_ENVIRONMENT_H

#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>
#include <stdexcept>

namespace poweruct {

    template<typename S>
    class Environment {

    public:

        virtual ~Environment() = default;

        virtual S getInitialState() = 0;

        virtual uint32_t getInitialObservation() = 0;

        virtual uint32_t getNumberOfObservations() = 0;

        virtual uint32_t getNumberOfActions() = 0;

        virtual std::vector<uint32_t> get_valid_actions(const S &) {
            std::vector<uint32_t> actions;
            uint32_t na = getNumberOfActions();
            actions.reserve(na);
            for (uint32_t action = 0; action < na; action++) {
                actions.emplace_back(action);
            }

            return actions;
        }

        virtual void seed(uint32_t s) = 0;

        virtual void reset() = 0;

        virtual void set_state(S state) = 0;

        virtual std::tuple<S, uint32_t, double, bool> step(uint32_t action) = 0;

        virtual std::tuple<uint32_t, S, uint32_t, double, bool> random_step() = 0;

        virtual std::tuple<S, uint32_t, double, bool> simulate(S state, uint32_t action) = 0;

        virtual std::tuple<uint32_t, S, uint32_t, double, bool> random_simulate(S state) = 0;

        virtual std::unique_ptr<std::vector<std::tuple<uint32_t, double, S, uint32_t >>> rollout(S state) = 0;

        virtual void render() = 0;

    };

}

#endif //C_ENVIRONMENT_H
