#ifndef C___SYSADMIN_RING_H
#define C___SYSADMIN_RING_H

#include <environment.h>
#include <random_utils.h>

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace poweruct {

    // Edit these compile-time constants to change the fixed environment size.
    constexpr uint32_t SYSADMIN_RING_NUM_COMPUTERS = 20;
    constexpr uint32_t SYSADMIN_RING_TIME_LIMIT = 50;

    struct SysAdminRingState {
        SysAdminRingState() : alive_mask(0), time(0) {}

        SysAdminRingState(uint32_t alive_mask, uint32_t time) : alive_mask(alive_mask), time(time) {}

        uint32_t alive_mask;
        uint32_t time;
    };

    class SysAdminRing : public Environment<SysAdminRingState> {

    public:
        SysAdminRing();

        ~SysAdminRing() override;

        SysAdminRingState getInitialState() override;

        uint32_t getInitialObservation() override;

        uint32_t getNumberOfObservations() override;

        uint32_t getNumberOfActions() override;

        void seed(uint32_t s) override;

        void reset() override;

        void set_state(SysAdminRingState state) override;

        std::tuple<SysAdminRingState, uint32_t, double, bool> step(uint32_t action) override;

        std::tuple<uint32_t, SysAdminRingState, uint32_t, double, bool> random_step() override;

        std::tuple<SysAdminRingState, uint32_t, double, bool> simulate(SysAdminRingState state,
                                                                        uint32_t action) override;

        std::tuple<uint32_t, SysAdminRingState, uint32_t, double, bool>
        random_simulate(SysAdminRingState state) override;

        std::unique_ptr<std::vector<std::tuple<uint32_t, double, SysAdminRingState, uint32_t >>>
        rollout(SysAdminRingState state) override;

        void render() override;

    private:
        bool initialized;
        uint32_t last_action;
        uint32_t num_observations;
        SysAdminRingState state;
        SysAdminRingState initial_state;
        RandomUtils r_util;
        std::vector<double> uniform_action_probs;

        uint32_t encode_observation(const SysAdminRingState &state) const;

        void validate_state(const SysAdminRingState &state) const;

        bool is_running(uint32_t alive_mask, uint32_t machine) const;

        uint32_t count_running(uint32_t alive_mask) const;

        double running_probability(bool prev_running, bool self_running) const;
    };

}

#endif //C___SYSADMIN_RING_H
