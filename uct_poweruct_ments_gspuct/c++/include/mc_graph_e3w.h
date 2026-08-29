#ifndef C___MC_GRAPH_E3W_H
#define C___MC_GRAPH_E3W_H

#include <graph_search.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace poweruct {

    /**
     * The convex regularizer Omega used at a decision node.
     *
     * All three share the same schema (E3W): the regularized value is the Legendre-Fenchel conjugate
     * Omega*(Q(s,.)/tau) scaled by tau, and the regularized policy is its gradient grad Omega*.
     *
     *      MaxEntropy      (MENTS)  Omega = -H(pi)          -> Omega* = log-sum-exp, grad = softmax
     *      RelativeEntropy (RENTS)  Omega = KL(pi||pi_par)  -> Omega* = log-sum-exp weighted by pi_par
     *      TsallisEntropy  (TENTS)  Omega = -H_2(pi)/2      -> Omega* = spmax,       grad = sparsemax
     */
    enum class RegularizerType {
        MaxEntropy,
        RelativeEntropy,
        TsallisEntropy
    };

    /**
     * Weighted log-sum-exp: Omega*(z) = log(sum_a w_a exp(z_a)), grad Omega*(z)_a ~ w_a exp(z_a).
     *
     * MENTS is the special case w_a = 1 (or w_a = 1/|A| once bias corrected); RENTS is w_a = pi_parent(a).
     * The max of z is factored out for numerical stability, exactly as in compute_action_weights.
     */
    inline double log_sum_exp_operator(const std::vector<double> &z, const std::vector<double> &w,
                                       std::vector<double> &policy) {
        policy.assign(z.size(), 0.);
        if (z.empty()) {
            return 0.;
        }

        double max_z = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < z.size(); i++) {
            if (w[i] > 0. && z[i] > max_z) {
                max_z = z[i];
            }
        }

        if (!std::isfinite(max_z)) {
            // Every action carries zero reference mass; fall back to uniform.
            double uniform = 1. / static_cast<double>(z.size());
            policy.assign(z.size(), uniform);
            return 0.;
        }

        double acc = 0.;
        for (std::size_t i = 0; i < z.size(); i++) {
            double weight = w[i] * std::exp(z[i] - max_z);
            if (!std::isfinite(weight)) {
                weight = 0.;
            }
            policy[i] = weight;
            acc += weight;
        }

        if (acc <= 0.) {
            double uniform = 1. / static_cast<double>(z.size());
            policy.assign(z.size(), uniform);
            return max_z;
        }

        for (std::size_t i = 0; i < policy.size(); i++) {
            policy[i] /= acc;
        }

        return max_z + std::log(acc);
    }

    /**
     * Sparsemax / spmax: the Tsallis (TENTS) conjugate.
     *
     *      K       = { a : 1 + i * z_(i) > sum_{j <= i} z_(j) }, z sorted descending
     *      c       = (sum_{a in K} z_a - 1) / |K|
     *      policy  = max(0, z_a - c)                             (grad Omega*, exactly sparse)
     *      Omega*  = sum_{a in K} (z_a^2 - c^2) / 2 + 1/2
     *
     * When 'bias_correct' the constant-vector bias (1 - 1/|K|)/2 is removed, so that
     * min_a z_a <= Omega*(z) <= max_a z_a. See the class comment for why that matters on a graph.
     */
    inline double sparsemax_operator(const std::vector<double> &z, std::vector<double> &policy, bool bias_correct) {
        policy.assign(z.size(), 0.);
        if (z.empty()) {
            return 0.;
        }

        std::vector<double> sorted(z);
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());

        std::size_t k = 0;
        double running_sum = 0.;
        double sparse_sum = 0.;
        for (std::size_t i = 0; i < sorted.size(); i++) {
            running_sum += sorted[i];
            if (1. + static_cast<double>(i + 1) * sorted[i] > running_sum) {
                k = i + 1;
                sparse_sum = running_sum;
            }
        }

        if (k == 0) {
            double uniform = 1. / static_cast<double>(z.size());
            policy.assign(z.size(), uniform);
            return *std::max_element(z.begin(), z.end());
        }

        double threshold = (sparse_sum - 1.) / static_cast<double>(k);

        double total = 0.;
        for (std::size_t i = 0; i < z.size(); i++) {
            double mass = z[i] - threshold;
            policy[i] = mass > 0. ? mass : 0.;
            total += policy[i];
        }

        if (total <= 0.) {
            double uniform = 1. / static_cast<double>(z.size());
            policy.assign(z.size(), uniform);
        } else {
            for (std::size_t i = 0; i < policy.size(); i++) {
                policy[i] /= total;
            }
        }

        double value = 0.5;
        for (std::size_t i = 0; i < k; i++) {
            value += (sorted[i] * sorted[i] - threshold * threshold) / 2.;
        }

        if (bias_correct) {
            value -= 0.5 * (1. - 1. / static_cast<double>(k));
        }

        return value;
    }

    struct E3WGraphEdgeStats {
        double q_value = 0.;
        uint32_t visits = 0;
        double reward_sum = 0.;
        // Successor counts N(s,a,s'), used to recompute Q(s,a) as a fresh expectation over current V(s').
        // Terminal transitions are deliberately omitted: they contribute a future value of zero.
        std::map<GraphStateKey, uint32_t> successors;
    };

    struct E3WGraphNodeStats {
        double value = 0.;
        uint32_t visits = 0;
        bool evaluated = false;
        std::vector<uint32_t> valid_actions;
        std::map<uint32_t, E3WGraphEdgeStats> edges;
    };

    struct E3WGraphTransition {
        E3WGraphTransition(const GraphStateKey &source, uint32_t action, const GraphStateKey &target, double reward,
                           bool terminal, std::vector<double> reference)
                : source(source), action(action), target(target), reward(reward), terminal(terminal),
                  reference(std::move(reference)) {}

        GraphStateKey source;
        uint32_t action;
        GraphStateKey target;
        double reward;
        bool terminal;
        // The reference distribution the source node used on this trial, aligned with its valid_actions.
        // Only RENTS reads it back; it is what replaces THTS++'s ThtsEnvContext parent-distribution passing.
        std::vector<double> reference;
    };

    /**
     * Convex-regularized Monte-Carlo *Graph* Search: the MENTS / RENTS / TENTS family transplanted from
     * a tree onto the transposition graph used by MCGraphSearchTree.
     *
     * Two things change relative to the tree formulation, and both are forced by the graph structure:
     *
     * 1. Q(s,a) is recomputed, not averaged. A tree node has one parent, so a running average of
     *    r + gamma * V(s') is a fine estimator. On a graph V(s') keeps being revised by *other* parents'
     *    trials, so a running average bakes in stale successor values. Instead we keep N(s,a,s') and
     *    R_sum(s,a) and recompute
     *        Q(s,a) = R_sum(s,a)/N(s,a) + gamma * sum_{s'} N(s,a,s')/N(s,a) * V(s')
     *    on every backup. This is exactly MentsCNode::backup_soft's visit-weighted average, except the
     *    successor set is shared across all parents of s'.
     *
     * 2. The operator must be non-expansive. In Full mode the key drops the timestep, so the graph may
     *    contain cycles, and this suite runs with discount 1.0. Raw log-sum-exp obeys
     *    V(s) >= max_a Q(s,a) + tau*log|A| on ties, so a zero-reward cycle s -> s' -> s is a fixed-point
     *    iteration V <- V + tau*log|A| that diverges. Removing each operator's constant-vector bias
     *    (normalising the reference weights for MENTS/RENTS, subtracting (1-1/|K|)/2 for TENTS) yields
     *    min_a Q <= V <= max_a Q, which is a non-expansion and cannot inflate around a cycle. This is
     *    the mellowmax correction, and it is on by default; 'bias_correct=false' recovers the textbook
     *    tree operators for Depth mode, where the key includes the timestep and the graph is a DAG.
     *
     * RENTS additionally needs its parent's action distribution. A graph node has many parents, so the
     * distribution cannot live on the node; it is carried along the sampled path (in E3WGraphTransition)
     * exactly as THTS++ threads it through a per-trial ThtsEnvContext.
     */
    template<typename S>
    class MCGraphE3WSearchTree : public AbstractSearchTree<S> {

    public:
        MCGraphE3WSearchTree(std::shared_ptr<Environment<S>> env, S initial_state, uint32_t initial_observation,
                             double discount_factor, double tau, double epsilon, RegularizerType regularizer,
                             GraphSearchMode mode, bool bias_correct = true, double max_explore_prob = 1.0,
                             double default_q_value = 0.0)
                : env(std::move(env)), na(this->env->getNumberOfActions()), discount_factor(discount_factor),
                  tau(tau), epsilon(epsilon), regularizer(regularizer), mode(mode), bias_correct(bias_correct),
                  max_explore_prob(max_explore_prob), default_q_value(default_q_value),
                  random_utils(std::make_shared<RandomUtils>()), current_state(initial_state) {
            (void) initial_observation;

            if (tau <= 0.) {
                throw std::runtime_error("MCGraphE3WSearchTree requires tau > 0");
            }
            if (epsilon < 0.) {
                throw std::runtime_error("MCGraphE3WSearchTree requires epsilon >= 0");
            }

            ensure_node(current_state);
        }

        void progress_tree(uint32_t action, S next_state, uint32_t next_obs) override {
            (void) action;
            (void) next_obs;
            current_state = next_state;
            ensure_node(current_state);
        }

        uint32_t search(uint32_t n_runs) override {
            ensure_node(current_state);
            for (uint32_t i = 0; i < n_runs; i++) {
                run_rollout();
            }

            return select_greedy_action();
        }

        void seed(uint32_t s) override {
            random_utils->seed(s);
        }

        std::vector<GraphActionStats> get_current_action_stats() const {
            std::vector<GraphActionStats> stats;
            stats.reserve(na);

            auto current_key = build_key(current_state);
            auto node_it = graph.find(current_key);
            for (uint32_t action = 0; action < na; action++) {
                GraphActionStats action_stats;
                action_stats.action = action;

                if (node_it != graph.end()) {
                    auto edge_it = node_it->second.edges.find(action);
                    if (edge_it != node_it->second.edges.end()) {
                        action_stats.q_value = edge_it->second.q_value;
                        action_stats.visits = edge_it->second.visits;
                    }
                }

                stats.emplace_back(action_stats);
            }

            return stats;
        }

    private:
        std::shared_ptr<Environment<S>> env;
        uint32_t na;
        double discount_factor;
        double tau;
        double epsilon;
        RegularizerType regularizer;
        GraphSearchMode mode;
        bool bias_correct;
        double max_explore_prob;
        double default_q_value;
        std::shared_ptr<RandomUtils> random_utils;
        using E3WGraphNodeMap = std::unordered_map<GraphStateKey, E3WGraphNodeStats, GraphStateKeyHash>;
        E3WGraphNodeMap graph;
        S current_state;

        GraphStateKey build_key(const S &state) const {
            return build_graph_state_key(state, mode, env.get());
        }

        E3WGraphNodeStats &ensure_node(const S &state) {
            auto entry = graph.emplace(build_key(state), E3WGraphNodeStats{});
            if (entry.second) {
                entry.first->second.valid_actions = env->get_valid_actions(state);
            }
            return entry.first->second;
        }

        E3WGraphNodeStats &node_at(const GraphStateKey &key) {
            auto it = graph.find(key);
            if (it == graph.end()) {
                throw std::runtime_error("Missing graph node in MCGraphE3WSearchTree");
            }
            return it->second;
        }

        double node_value(const GraphStateKey &key) const {
            auto it = graph.find(key);
            return it == graph.end() ? 0. : it->second.value;
        }

        /** Q(s,a) as currently estimated, or the default for an edge nothing has traversed yet. */
        double edge_q_value(const E3WGraphNodeStats &node, uint32_t action) const {
            auto it = node.edges.find(action);
            if (it == node.edges.end() || it->second.visits == 0) {
                return default_q_value;
            }
            return it->second.q_value;
        }

        /**
         * The reference weights w_a fed to the operator, aligned with node.valid_actions.
         *
         *  - RENTS uses the parent's action distribution restricted to this node's valid actions and
         *    renormalised. A parent policy that gives every valid action here zero mass (possible with
         *    TENTS-style sparsity or a disjoint action set) degenerates to uniform rather than to a
         *    zero distribution.
         *  - MENTS/TENTS use uniform weights. For MENTS uniform-and-normalised is precisely the
         *    mellowmax bias correction; without correction the weights are all 1.
         */
        std::vector<double> build_reference(const E3WGraphNodeStats &node,
                                            const std::vector<double> &parent_policy) const {
            std::size_t n = node.valid_actions.size();
            std::vector<double> reference(n, 0.);

            if (regularizer == RegularizerType::RelativeEntropy && !parent_policy.empty()) {
                double total = 0.;
                for (std::size_t i = 0; i < n; i++) {
                    uint32_t action = node.valid_actions[i];
                    double mass = action < parent_policy.size() ? parent_policy[action] : 0.;
                    reference[i] = mass;
                    total += mass;
                }

                if (total > 1e-12) {
                    for (std::size_t i = 0; i < n; i++) {
                        reference[i] /= total;
                    }
                    return reference;
                }
            }

            double weight = (bias_correct || regularizer == RegularizerType::RelativeEntropy)
                            ? 1. / static_cast<double>(n)
                            : 1.;
            reference.assign(n, weight);
            return reference;
        }

        /**
         * Evaluates the operator at a node: returns tau * Omega*(Q/tau) and fills the regularized policy.
         */
        double evaluate_operator(const E3WGraphNodeStats &node, const std::vector<double> &reference,
                                 std::vector<double> &policy) const {
            std::size_t n = node.valid_actions.size();
            std::vector<double> z(n, 0.);
            for (std::size_t i = 0; i < n; i++) {
                z[i] = edge_q_value(node, node.valid_actions[i]) / tau;
            }

            double value_over_tau;
            if (regularizer == RegularizerType::TsallisEntropy) {
                value_over_tau = sparsemax_operator(z, policy, bias_correct);
            } else {
                value_over_tau = log_sum_exp_operator(z, reference, policy);
            }

            return tau * value_over_tau;
        }

        /**
         * E3W sampling: pi(a|s) = (1 - lambda) * grad Omega*(Q/tau)_a + lambda / |A|, with the MENTS
         * schedule lambda = epsilon * |A| / log(N(s) + 1) clipped to [0, max_explore_prob].
         *
         * Note that on a graph N(s) pools visits from every parent of s, so lambda decays faster than in
         * a tree. That is the intended effect of transposition sharing, but it does mean the E3W
         * exploration guarantee is stated over the pooled count rather than a per-parent one.
         */
        uint32_t sample_action(const E3WGraphNodeStats &node, const std::vector<double> &policy) {
            const auto &valid_actions = node.valid_actions;
            double n_actions = static_cast<double>(valid_actions.size());
            double lambda = (epsilon * n_actions) / std::log(static_cast<double>(node.visits) + 1.);
            if (!std::isfinite(lambda) || lambda > max_explore_prob) {
                lambda = max_explore_prob;
            }

            if (random_utils->sample_uniform() < lambda) {
                return random_utils->sample_uniform(valid_actions);
            }

            std::vector<double> likelihoods(policy);
            return valid_actions[random_utils->sample_discrete(likelihoods)];
        }

        /** Recommendation is the largest regularized Q, matching MentsDNode::recommend_action_best_soft_value. */
        uint32_t select_greedy_action() {
            auto &node = ensure_node(current_state);
            if (node.valid_actions.empty()) {
                throw std::runtime_error("No valid actions available in MCGraphE3WSearchTree");
            }

            std::vector<uint32_t> candidates;
            double best_score = -std::numeric_limits<double>::infinity();
            bool saw_visited = false;

            for (uint32_t action : node.valid_actions) {
                auto it = node.edges.find(action);
                if (it == node.edges.end() || it->second.visits == 0) {
                    if (!saw_visited) {
                        candidates.emplace_back(action);
                    }
                    continue;
                }

                if (!saw_visited) {
                    saw_visited = true;
                    candidates.clear();
                }

                double score = it->second.q_value;
                if (score > best_score + 1e-12) {
                    best_score = score;
                    candidates.clear();
                    candidates.emplace_back(action);
                } else if (std::abs(score - best_score) <= 1e-12) {
                    candidates.emplace_back(action);
                }
            }

            if (candidates.empty()) {
                return random_utils->sample_uniform(node.valid_actions);
            }

            return random_utils->sample_uniform(candidates);
        }

        /** Discounted return of a random rollout, used once to seed a freshly created node's value. */
        double evaluate_leaf(const S &state) {
            auto trajectory = env->rollout(state);
            double value = 0.;
            double disc = 1.;
            for (const auto &step : *trajectory) {
                value += disc * std::get<1>(step);
                disc *= discount_factor;
            }
            return value;
        }

        void run_rollout() {
            std::vector<E3WGraphTransition> path;
            S sim_state = current_state;
            auto current_key = build_key(sim_state);
            std::vector<double> parent_policy;

            while (true) {
                auto &current_node = node_at(current_key);
                if (current_node.valid_actions.empty()) {
                    break;
                }

                auto reference = build_reference(current_node, parent_policy);
                std::vector<double> policy;
                evaluate_operator(current_node, reference, policy);
                uint32_t action = sample_action(current_node, policy);

                S next_state;
                uint32_t next_obs;
                double reward;
                bool done;
                std::tie(next_state, next_obs, reward, done) = env->simulate(sim_state, action);

                // Re-index the policy by action id so the child can read it as its parent distribution.
                std::vector<double> policy_by_action(na, 0.);
                for (std::size_t i = 0; i < current_node.valid_actions.size(); i++) {
                    policy_by_action[current_node.valid_actions[i]] = policy[i];
                }

                auto next_key = build_key(next_state);
                path.emplace_back(current_key, action, next_key, reward, done, reference);

                if (done) {
                    break;
                }

                bool fresh = graph.find(next_key) == graph.end();
                auto &next_node = ensure_node(next_state);
                if (fresh || !next_node.evaluated) {
                    next_node.value = evaluate_leaf(next_state);
                    next_node.evaluated = true;
                    break;
                }

                sim_state = next_state;
                current_key = next_key;
                parent_policy = std::move(policy_by_action);
            }

            backup_path(path);
        }

        /**
         * Bottom-up soft backup along the sampled path.
         *
         * Each step refreshes Q(s,a) from the *current* successor values (see class comment), then
         * recomputes V(s) with the regularizer, reusing the same reference distribution the trial used
         * on the way down so that RENTS backs up against the policy it actually sampled from.
         */
        void backup_path(const std::vector<E3WGraphTransition> &path) {
            if (path.empty()) {
                return;
            }

            auto leaf_it = graph.find(path.back().target);
            if (leaf_it != graph.end()) {
                leaf_it->second.visits += 1;
            }

            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                auto &source_node = node_at(it->source);
                auto &edge = source_node.edges[it->action];

                edge.visits += 1;
                edge.reward_sum += it->reward;
                if (!it->terminal) {
                    edge.successors[it->target] += 1;
                }

                double expected_future = 0.;
                for (const auto &successor : edge.successors) {
                    expected_future += static_cast<double>(successor.second) * node_value(successor.first);
                }
                expected_future /= static_cast<double>(edge.visits);

                edge.q_value = edge.reward_sum / static_cast<double>(edge.visits) +
                               discount_factor * expected_future;

                source_node.visits += 1;
                std::vector<double> policy;
                source_node.value = evaluate_operator(source_node, it->reference, policy);
                source_node.evaluated = true;
            }
        }
    };

}

#endif //C___MC_GRAPH_E3W_H
