#ifndef C___FROZEN_LAKE_H
#define C___FROZEN_LAKE_H

#include <discrete_environment.h>
#include <iostream>

namespace poweruct {

    class FrozenLake : public DiscreteEnvironment {

    public:
        FrozenLake(bool slippery, bool use_rollout_heuristic);
        ~FrozenLake() override;

        void render() override;
        DiscreteEnvironmentState getInitialState() override;
        uint32_t getInitialObservation() override;

    protected:
        RandomUtils &get_r_util() override;
        std::vector<double>& get_rollout_action_probs(DiscreteEnvironmentState &state) override;

    private:
        bool use_rollout_heuristic;

    };

}

#endif //C___FROZEN_LAKE_H
