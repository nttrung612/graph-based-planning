#include <utility>

#ifndef C___TREE_SEARCH_NODES_H
#define C___TREE_SEARCH_NODES_H

#include <rollout_container.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <iostream>
#include <limits>
#include <functional>
#include <tree_search.h>
#include <random_utils.h>

namespace poweruct {

    template<typename S, typename I>
    class QNode;

    template<typename S, typename I>
    class RolloutContainer;

    template<typename S>
    class PowerUCTNode;

    template<typename S, typename I>
    class VNode {

    public:
        VNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
              std::shared_ptr<RandomUtils> random_utils,
              std::function<std::vector<uint32_t>(const S &)> valid_action_provider) :
                state(state), observation(observation), na(na), parent(), final(final), expanded(false),
                random_utils(std::move(random_utils)), discount_factor(discount_factor),
                valid_action_provider(std::move(valid_action_provider)),
                valid_actions(this->valid_action_provider(this->state)),
                rollout_container(na, discount_factor, std::bind(&VNode<S, I>::node_constructor, this,
                                                                 std::placeholders::_1, std::placeholders::_2,
                                                                 std::placeholders::_3)) {};

        ~VNode() = default;

        S get_state() {
            return state;
        }

        uint32_t get_observation() {
            return observation;
        }

        void set_parent(std::shared_ptr<QNode<S, I>> parent) {
            this->parent = parent;
        }

        std::shared_ptr<QNode<S, I>> get_parent() {
            return parent.lock();
        }

        bool has_parent() {
            return !parent.expired();
        }

        bool is_final() {
            return final;
        }

        bool is_expanded() {
            return expanded;
        }

        void
        store_rollout(std::unique_ptr<std::vector<std::tuple<uint32_t, double, S, uint32_t> > > trajectory) {
            if (expanded) {
                throw std::runtime_error("Only unexpanded nodes can be used to store rollouts");
            }

            rollout_container.store_rollout(std::move(trajectory));
        }

        void update(uint32_t action) {
            auto it = q_nodes.find(action);
            if (it == q_nodes.end()) {
                throw std::runtime_error("Unknown action in V-Node!");
            }

            q_stats[action] = it->second->get_q();
        }

        uint32_t select_greedy_action() {
            if (valid_actions.empty()) {
                throw std::runtime_error("No valid actions available in V-Node");
            }

            double max_q = -std::numeric_limits<double>::infinity();
            std::vector<uint32_t> candidates;
            for (uint32_t action : valid_actions) {
                auto it = q_stats.find(action);
                if (it == q_stats.end()) {
                    continue;
                }

                double q_value = std::get<0>(it->second);
                if (q_value > max_q + 1e-12) {
                    max_q = q_value;
                    candidates.clear();
                    candidates.emplace_back(action);
                } else if (std::abs(q_value - max_q) <= 1e-12) {
                    candidates.emplace_back(action);
                }
            }

            if (candidates.empty()) {
                return valid_actions.front();
            }

            return random_utils->sample_uniform(candidates);
        }

        virtual uint32_t select_action() = 0;

        std::shared_ptr<QNode<S, I>> get_child(uint32_t a) {
            if (!expanded) {
                throw std::runtime_error("Get child called on an unexpanded node!");
            }

            auto it = q_nodes.find(a);
            if (it == q_nodes.end()) {
                return std::shared_ptr<QNode<S, I>>(nullptr);
            }

            return it->second;
        }

        void set_children(uint32_t a, std::shared_ptr<QNode<S, I>> q_node) {
            q_nodes[a] = q_node;
        }

        virtual std::tuple<double, uint32_t> get_value() = 0;

        void expand(const std::shared_ptr<VNode<S, I>> &self_reference) {
            if (expanded || final) {
                throw std::runtime_error("Expand cannot be called on an already expanded or final V-Node");
            }

            // This will produce an error if it is called twice
            auto q_nodes = rollout_container.expand();
            for (auto &&kv : q_nodes) {
                this->set_children(kv.first, kv.second);
                kv.second->set_parent(std::shared_ptr<VNode<S, I>>(self_reference));
                this->update(kv.first);
            }

            expanded = true;
        }

    protected:
        S state;
        uint32_t observation;
        uint32_t na;
        std::weak_ptr<QNode<S, I>> parent;
        bool expanded;
        std::shared_ptr<RandomUtils> random_utils;
        double discount_factor;
        std::function<std::vector<uint32_t>(const S &)> valid_action_provider;
        std::vector<uint32_t> valid_actions;

        std::map<uint32_t, std::tuple<double, uint32_t >> q_stats;
        std::map<uint32_t, std::shared_ptr<QNode<S, I>>> q_nodes;

        bool final;
        RolloutContainer<S, I> rollout_container;

        virtual std::shared_ptr<I> node_constructor(S s, uint32_t obs, bool done) = 0;

        double get_pre_exp_reward() {
            return rollout_container.get_total_reward();
        };

        uint32_t get_pre_exp_visits() {
            return rollout_container.get_total_visits();
        };
    };

    template<typename S, typename I>
    class QNode {

    public:
        QNode(uint32_t action, double discount_factor) : action(action), discount_factor(discount_factor),
                                                         immediate_rewards(0.) {};

        ~QNode() = default;

        void store_reward(double reward) {
            immediate_rewards += reward;
        }

        void update(uint32_t next_obs) {
            auto it = v_nodes.find(next_obs);
            if (it == v_nodes.end()) {
                throw std::runtime_error("Unknown next state in Q Node!");
            }

            std::tie(values[next_obs], visits[next_obs]) = it->second->get_value();
        }

        std::tuple<double, uint32_t> get_q() {
            double total_disc_rew = 0.;
            uint32_t total_visits = 0;

            for (auto &&kv : values) {
                uint32_t cur_visits = visits[kv.first];
                total_disc_rew += kv.second * cur_visits;
                total_visits += cur_visits;
            }

            return std::make_tuple((immediate_rewards + discount_factor * total_disc_rew) / ((double) total_visits),
                                   total_visits);
        }

        std::shared_ptr<VNode<S, I>> get_children(uint32_t obs) {
            auto it = v_nodes.find(obs);
            if (it == v_nodes.end()) {
                return std::shared_ptr<VNode<S, I>>(nullptr);
            }

            return it->second;
        }

        void set_children(uint32_t obs, std::shared_ptr<VNode<S, I>> v_node) {
            v_nodes[obs] = v_node;
        }

        void set_parent(std::shared_ptr<VNode<S, I>> parent) {
            this->parent = parent;
        }

        std::shared_ptr<VNode<S, I>> get_parent() {
            return parent.lock();
        }

        uint32_t get_action() {
            return action;
        }

    private:
        std::map<uint32_t, double> values;
        std::map<uint32_t, uint32_t> visits;
        std::map<uint32_t, std::shared_ptr<VNode<S, I>>> v_nodes;
        std::weak_ptr<VNode<S, I>> parent;
        uint32_t action;

        double immediate_rewards;
        double discount_factor;

    };

}

#endif //C___TREE_SEARCH_NODES_H
