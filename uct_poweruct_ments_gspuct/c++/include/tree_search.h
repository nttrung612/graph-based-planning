#ifndef C___TREE_SEARCH_H
#define C___TREE_SEARCH_H

#include <tree_search_nodes.h>
#include <environment.h>
#include <memory>
#include <random_utils.h>
#include <cmath>
#include <util.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace poweruct {

    template<typename S, typename I>
    class UCTNode : public VNode<S, I> {

    public:
        UCTNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                std::shared_ptr<RandomUtils> random_utils,
                std::function<std::vector<uint32_t>(const S &)> valid_action_provider, double alpha) :
                VNode<S, I>(state, observation, discount_factor, na, final, random_utils,
                            std::move(valid_action_provider)), alpha(alpha) {};

        uint32_t select_action() override {
            if (this->valid_actions.empty()) {
                throw std::runtime_error("No valid actions available in UCTNode");
            }

            uint32_t total_visits = 0;
            for (uint32_t action : this->valid_actions) {
                auto child = VNode<S, I>::q_stats.find(action);
                if (child != VNode<S, I>::q_stats.end()) {
                    total_visits += std::get<1>(child->second);
                }
            }

            std::vector<uint32_t> action_candidates;
            double max_q = -std::numeric_limits<double>::infinity();
            bool saw_unvisited = false;
            for (uint32_t action : this->valid_actions) {
                auto child = VNode<S, I>::q_stats.find(action);
                if (child == VNode<S, I>::q_stats.end() || std::get<1>(child->second) == 0) {
                    if (!saw_unvisited) {
                        saw_unvisited = true;
                        action_candidates.clear();
                    }
                    action_candidates.emplace_back(action);
                    continue;
                }

                if (saw_unvisited) {
                    continue;
                }

                double exploration_bonus = alpha * sqrt(
                        log(static_cast<double>(std::max<uint32_t>(1, total_visits))) /
                        static_cast<double>(std::get<1>(child->second)) + 1e-10);
                double q = std::get<0>(child->second) + exploration_bonus;
                if (q >= max_q) {
                    if (q > max_q) {
                        action_candidates.clear();
                        max_q = q;
                    }
                    action_candidates.emplace_back(action);
                }
            }

            return VNode<S, I>::random_utils->sample_uniform(action_candidates);
        };

    protected:
        double alpha;

    };

    template<typename S>
    class MaxUCTNode : public UCTNode<S, MaxUCTNode<S>> {

    public:
        MaxUCTNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                   std::shared_ptr<RandomUtils> random_utils,
                   std::function<std::vector<uint32_t>(const S &)> valid_action_provider, double alpha) :
                UCTNode<S, MaxUCTNode<S>>(state, observation, discount_factor, na, final, random_utils,
                                          std::move(valid_action_provider), alpha) {};

        virtual std::tuple<double, uint32_t> get_value() {
            uint32_t n_total = this->get_pre_exp_visits();

            double max_avg;
            if (this->q_stats.size() == 0) {
                max_avg = 0.;
            } else {
                max_avg = -1e6;
                for (auto &&kv : this->q_stats) {
                    double avg_rew = std::get<0>(kv.second);
                    n_total += std::get<1>(kv.second);
                    if (avg_rew > max_avg) {
                        max_avg = avg_rew;
                    }
                }
            }

            return std::make_tuple((this->get_pre_exp_reward() / n_total) + max_avg, n_total);
        }

    protected:
        std::shared_ptr<MaxUCTNode<S>> node_constructor(S s, uint32_t obs, bool done) {
            return std::make_shared<MaxUCTNode<S>>(s, obs, this->discount_factor, this->na, done, this->random_utils,
                                                   this->valid_action_provider, this->alpha);
        }

    };

    template<typename S>
    class PowerUCTNode : public UCTNode<S, PowerUCTNode<S>> {

    public:
        PowerUCTNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                     std::shared_ptr<RandomUtils> random_utils,
                     std::function<std::vector<uint32_t>(const S &)> valid_action_provider, double alpha, double p) :
                UCTNode<S, PowerUCTNode<S>>(state, observation, discount_factor, na, final, random_utils,
                                            std::move(valid_action_provider), alpha),
                p(p) {};

        uint32_t select_action() override {
            if (this->valid_actions.empty()) {
                throw std::runtime_error("No valid actions available in PowerUCTNode");
            }

            uint32_t total_visits = 0;
            for (uint32_t action : this->valid_actions) {
                auto child = this->q_stats.find(action);
                if (child != this->q_stats.end()) {
                    total_visits += std::get<1>(child->second);
                }
            }

            const double total_visits_term = pow(static_cast<double>(total_visits), 0.25);
            std::vector<uint32_t> action_candidates;
            double max_q = -std::numeric_limits<double>::infinity();
            for (uint32_t action : this->valid_actions) {
                auto child = this->q_stats.find(action);
                double q;
                if (child == this->q_stats.end() || std::get<1>(child->second) == 0) {
                    q = std::numeric_limits<double>::infinity();
                } else {
                    double n = static_cast<double>(std::get<1>(child->second));
                    double exploration_bonus = this->alpha * (total_visits_term / sqrt(n));
                    q = std::get<0>(child->second) + exploration_bonus;
                }

                if (q >= max_q) {
                    if (q > max_q) {
                        action_candidates.clear();
                        max_q = q;
                    }
                    action_candidates.emplace_back(action);
                }
            }

            return this->random_utils->sample_uniform(action_candidates);
        }

        virtual std::tuple<double, uint32_t> get_value() {
            uint32_t n_total = this->get_pre_exp_visits();
            for (auto &&kv : this->q_stats) {
                n_total += std::get<1>(kv.second);
            }

            if (p == 0.) {
                double max_avg = 0.;
                bool saw_visited = false;
                for (auto &&kv : this->q_stats) {
                    if (std::get<1>(kv.second) == 0) {
                        continue;
                    }

                    double avg_rew = std::get<0>(kv.second);
                    if (!saw_visited || avg_rew > max_avg) {
                        max_avg = avg_rew;
                        saw_visited = true;
                    }
                }

                return std::make_tuple((this->get_pre_exp_reward() / n_total) + max_avg, n_total);
            }

            double total = 0.;
            for (auto &&kv : this->q_stats) {
                double q = std::get<0>(kv.second);
                double n = (double) std::get<1>(kv.second);
                total += (n / ((double) n_total)) * pow(q, p);
            }

            double value = (this->get_pre_exp_reward() / n_total) + pow(total, 1. / p);
            return std::make_tuple(value, n_total);
        }

    protected:
        std::shared_ptr<PowerUCTNode<S>> node_constructor(S s, uint32_t obs, bool done) {
            return std::make_shared<PowerUCTNode<S>>(s, obs, this->discount_factor, this->na, done,
                                                     this->random_utils, this->valid_action_provider, this->alpha,
                                                     this->p);
        }

    private:
        double p;
    };

    template<typename S>
    class MaxEntropyNode : public VNode<S, MaxEntropyNode<S>> {
    public:
        MaxEntropyNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                       std::shared_ptr<RandomUtils> random_utils,
                       std::function<std::vector<uint32_t>(const S &)> valid_action_provider, float tau,
                       float epsilon) :
                VNode<S, MaxEntropyNode<S>>(state, observation, discount_factor, na, final, random_utils,
                                            std::move(valid_action_provider)),
                tau(tau), epsilon(epsilon) {}

        uint32_t select_action() {
            if (this->valid_actions.empty()) {
                throw std::runtime_error("No valid actions available in MaxEntropyNode");
            }

            std::vector<double> q_values(this->valid_actions.size(), 1e6);
            uint32_t total_visits = 0;
            for (uint32_t i = 0; i < this->valid_actions.size(); i++) {
                auto child = this->q_stats.find(this->valid_actions[i]);
                if (child != this->q_stats.end()) {
                    q_values[i] = std::get<0>(child->second);
                    total_visits += std::get<1>(child->second);
                }
            }

            double v_soft = compute_soft_value(q_values);
            std::vector<double> likelihoods;
            compute_likelihoods(v_soft, q_values, likelihoods);

            double lambda = (epsilon * static_cast<double>(this->valid_actions.size())) /
                            log(static_cast<double>(total_visits) + 1.);
            if (this->random_utils->sample_uniform() > lambda) {
                return this->valid_actions[this->random_utils->sample_discrete(likelihoods)];
            } else {
                return this->random_utils->sample_uniform(this->valid_actions);
            }
        }

        std::tuple<double, uint32_t> get_value() {
            static std::vector<double> q_values;
            q_values.clear();

            uint32_t total_visits = this->get_pre_exp_visits();
            for (auto const &child : this->q_stats) {
                q_values.push_back(std::get<0>(child.second));
                total_visits += std::get<1>(child.second);
            }

            if (q_values.empty()) {
                return std::make_tuple(this->get_pre_exp_reward() / total_visits, total_visits);
            } else {
                double value = (this->get_pre_exp_reward() / total_visits) + compute_soft_value(q_values);
                return std::make_tuple(value, total_visits);
            }
        }

    protected:
        std::shared_ptr<MaxEntropyNode<S>> node_constructor(S s, uint32_t obs, bool done) {
            return std::make_shared<MaxEntropyNode<S>>(s, obs, this->discount_factor, this->na, done,
                                                       this->random_utils, this->valid_action_provider, this->tau,
                                                       this->epsilon);
        };

    private:
        double tau;
        double epsilon;

        void compute_likelihoods(double v_soft, const std::vector<double> &q_values, std::vector<double> &likelihoods) {
            static std::vector<double> advs;
            advs.clear();

            for (auto q_value : q_values) {
                advs.push_back(q_value - v_soft);
            }
            double max_adv = *max_element(advs.begin(), advs.end());

            likelihoods.clear();
            double total_likelihood = 0.;
            for (auto adv : advs) {
                double unnormalized_likelihood = exp((adv - max_adv) / this->tau);
                likelihoods.push_back(unnormalized_likelihood);
                total_likelihood += unnormalized_likelihood;
            }

            for (uint32_t i = 0; i < likelihoods.size(); i++) {
                likelihoods[i] /= total_likelihood;
            }
        }

        double compute_soft_value(const std::vector<double> &q_values) {
            double max_q_value = *max_element(q_values.begin(), q_values.end());
            double acc = 0.;
            for (double q_value : q_values) {
                acc += exp((q_value - max_q_value) / this->tau);
            }

            return max_q_value + this->tau * log(acc);
        }
    };

    template<typename S>
    class AbstractSearchTree {

    public:
        virtual void progress_tree(uint32_t action, S next_state, uint32_t next_obs) = 0;

        virtual uint32_t search(uint32_t n_runs) = 0;

        virtual void seed(uint32_t s) = 0;

    };

    template<typename S, typename I, typename ... Types>
    class SearchTree : public AbstractSearchTree<S> {

    public:
        SearchTree(std::shared_ptr<Environment<S>> env, S initial_state, uint32_t initial_obs, double discount_factor,
                   Types... extra_args) :
                env(env), na(env->getNumberOfActions()), discount_factor(discount_factor),
                random_utils(std::make_shared<RandomUtils>()),
                valid_action_provider([env](const S &state) {
                    return env->get_valid_actions(state);
                }),
                node_constructor([this, extra_args...](S state, uint32_t observation, bool final) {
                    return std::make_shared<I>(state, observation, this->discount_factor, this->na, final,
                                               this->random_utils, this->valid_action_provider, extra_args...);
                }) {
            root = node_constructor(initial_state, initial_obs, false);
        }

        ~SearchTree() = default;

        void progress_tree(uint32_t action, S next_state, uint32_t next_obs) {
            std::shared_ptr<VNode<S, I>> v_child;
            if (root->is_expanded()) {
                auto q_child = root->get_child(action);
                if (q_child) {
                    v_child = q_child->get_children(next_obs);
                }
            }

            if (!v_child) {
                root = node_constructor(next_state, next_obs, false);
            } else {
                root = v_child;
            }
        }

        uint32_t search(uint32_t n_runs) {
            for (uint32_t i = 0; i < n_runs; i++) {
                bool run = true;
                bool is_expand = false;
                std::shared_ptr<VNode<S, I>> cur_node = root;
                while (run) {
                    if (cur_node->is_final()) {
                        cur_node->store_rollout(
                                make_unique<std::vector<std::tuple<uint32_t, double, S, uint32_t>>>());
                        run = false;
                    } else if (!cur_node->is_expanded()) {
                        cur_node->expand(cur_node);
                        is_expand = true;
                    } else {
                        S next_state;
                        uint32_t next_obs, action;
                        double reward;
                        bool done;
                        if (is_expand) {
                            std::tie(action, next_state, next_obs, reward, done) = env->random_simulate(
                                    cur_node->get_state());
                        } else {
                            action = cur_node->select_action();
                            std::tie(next_state, next_obs, reward, done) = env->simulate(cur_node->get_state(), action);
                        }

                        auto q_child = cur_node->get_child(action);
                        if (!q_child) {
                            q_child = std::make_shared<QNode<S, I>>(action, discount_factor);
                            cur_node->set_children(action, q_child);
                            q_child->set_parent(cur_node);
                        }

                        auto v_child = q_child->get_children(next_obs);

                        if (!v_child) {
                            std::unique_ptr<std::vector<std::tuple<uint32_t, double, S, uint32_t >>> remaining_reward;
                            if (done) {
                                remaining_reward = make_unique<std::vector<std::tuple<uint32_t, double, S, uint32_t>>>();
                            } else {
                                remaining_reward = env->rollout(next_state);
                            }
                            v_child = node_constructor(next_state, next_obs, done);
                            v_child->store_rollout(std::move(remaining_reward));

                            q_child->set_children(next_obs, v_child);
                            v_child->set_parent(q_child);
                            run = false;
                        }

                        q_child->store_reward(reward);
                        cur_node = v_child;
                    }
                }

                while (cur_node->has_parent()) {
                    auto q_parent = cur_node->get_parent();
                    q_parent->update(cur_node->get_observation());
                    auto v_parent = q_parent->get_parent();
                    v_parent->update(q_parent->get_action());
                    cur_node = v_parent;
                }
            }

            return root->select_greedy_action();
        }

        void seed(uint32_t s) {
            random_utils->seed(s);
        }

    private:
        std::shared_ptr<Environment<S>> env;
        uint32_t na;
        double discount_factor;
        std::shared_ptr<RandomUtils> random_utils;
        std::function<std::vector<uint32_t>(const S &)> valid_action_provider;
        std::shared_ptr<VNode<S, I>> root;
        std::function<std::shared_ptr<I>(S, uint32_t, bool)> node_constructor;

    };

    template<typename S>
    using MaxUCTSearchTree = SearchTree<S, MaxUCTNode<S>, double>;

    template<typename S>
    using PowerUCTSearchTree = SearchTree<S, PowerUCTNode<S>, double, double>;

    template<typename S>
    using MaxEntropySearchTree = SearchTree<S, MaxEntropyNode<S>, double, double>;

}

#endif //C___TREE_SEARCH_H
