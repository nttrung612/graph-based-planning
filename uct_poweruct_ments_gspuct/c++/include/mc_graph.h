#ifndef C___MC_GRAPH_H
#define C___MC_GRAPH_H

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

    struct MCGraphEdgeStats {
        double q_value = 0.;
        uint32_t visits = 0;
        // Successor counts T^{s'}_{s,h,a}: how often this edge realized each distinct successor node.
        // Only maintained when the ER child channel is active (c3 > 0), so the plain GS-Power-UCT
        // baseline keeps its original memory profile. Terminal transitions are counted like any
        // other, because this is the empirical support of p_t(.|s,h,a) and not a value: the sum over
        // s' equals `visits`, which is what the envelope B^ER <= (C2 + C3 M)/T_{s,h,a} relies on.
        std::map<GraphStateKey, uint32_t> successors;
    };

    struct MCGraphNodeStats {
        double value = 0.;
        uint32_t visits = 0;
        std::map<uint32_t, MCGraphEdgeStats> edges;
    };

    struct MCGraphTransition {
        MCGraphTransition(const GraphStateKey &source, uint32_t action, const GraphStateKey &target, double reward,
                          bool terminal)
                : source(source), action(action), target(target), reward(reward), terminal(terminal) {}

        GraphStateKey source;
        uint32_t action;
        GraphStateKey target;
        double reward;
        bool terminal;
    };

    /**
     * Graph-based stochastic Power-UCT, optionally with the effective-resistance bonus
     * (GS-Power-UCT-ER).
     *
     * With c2 = c3 = 0 this is exactly the unperturbed base algorithm: selection by
     *
     *      argmax_a  Qhat(s,h,a) + C1 * T(s,h)^{b/beta} / T(s,h,a)^{alpha/beta}
     *
     * at the optimal tuning alpha/beta = 1/2, b/beta = 1/4 (hence N^{0.25}/sqrt(n)), a power-mean
     * backup of order p, and argmax Qhat as the recommendation.
     *
     * With c2 or c3 positive, the selection rule additionally pays the one-step truncation of the
     * recursive resistance Rtilde -- see er_bonus(). Nothing else changes: the backup, the stored
     * estimates and the recommended action stay unperturbed, so the algorithm targets the same
     * fixed point and the bonus only redirects *which* arms get pulled.
     *
     * ER is available in Depth mode only, and the constructor enforces it. Rtilde is defined by a
     * topological recursion over a layered DAG: Depth mode keys carry the timestep, so every edge
     * advances h by one and the graph is acyclic by construction. Full mode merges a state across
     * depths and can close cycles, where the recursion has no fixed point and the resistance
     * reading of the bonus collapses (the truncated bonus would still evaluate, since it only
     * reads counts, but it would no longer mean what it is derived to mean). Cross-depth merging
     * needs spectral substitutes instead, which are outside this implementation.
     */
    template<typename S>
    class MCGraphSearchTree : public AbstractSearchTree<S> {

    public:
        MCGraphSearchTree(std::shared_ptr<Environment<S>> env, S initial_state, uint32_t initial_observation,
                          double discount_factor, double c, double p, GraphSearchMode mode,
                          double c2 = 0., double c3 = 0.)
                : env(std::move(env)), na(this->env->getNumberOfActions()), discount_factor(discount_factor), c(c),
                  p(p), mode(mode), c2(c2), c3(c3), random_utils(std::make_shared<RandomUtils>()),
                  current_state(initial_state) {
            (void) initial_observation;

            if (p < 0.) {
                throw std::runtime_error("MCGraphSearchTree requires p >= 0");
            }
            if (c < 0.) {
                throw std::runtime_error("MCGraphSearchTree requires c >= 0");
            }
            if (c2 < 0. || c3 < 0.) {
                throw std::runtime_error("MCGraphSearchTree requires c2 >= 0 and c3 >= 0");
            }
            if ((c2 > 0. || c3 > 0.) && mode != GraphSearchMode::Depth) {
                throw std::runtime_error("The effective-resistance bonus requires the layered "
                                         "depth-augmented graph (GraphSearchMode::Depth); the "
                                         "recursive resistance is undefined under cross-depth merging");
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
        double c;
        double p;
        GraphSearchMode mode;
        double c2;
        double c3;
        std::shared_ptr<RandomUtils> random_utils;
        using MCGraphNodeMap = std::unordered_map<GraphStateKey, MCGraphNodeStats, GraphStateKeyHash>;
        MCGraphNodeMap graph;
        S current_state;

        GraphStateKey build_key(const S &state) const {
            return build_graph_state_key(state, mode, env.get());
        }

        double effective_discount() const {
            return discount_factor - ((mode == GraphSearchMode::Depth) ? 0.0 : 0.0);
        }

        MCGraphNodeStats &ensure_node(const S &state) {
            return ensure_node(build_key(state));
        }

        MCGraphNodeStats &ensure_node(const GraphStateKey &key) {
            return ensure_node_entry(key).first->second;
        }

        std::pair<typename MCGraphNodeMap::iterator, bool> ensure_node_entry(const GraphStateKey &key) {
            return graph.emplace(key, MCGraphNodeStats{});
        }

        MCGraphEdgeStats &ensure_edge(const GraphStateKey &key, uint32_t action) {
            auto &node = ensure_node(key);
            return node.edges[action];
        }

        /** Pooled visit count T_{s,h} of a node, from all incoming paths. */
        uint32_t node_visits(const GraphStateKey &key) const {
            auto it = graph.find(key);
            return it == graph.end() ? 0 : it->second.visits;
        }

        /**
         * The effective-resistance exploration bonus: the one-step truncation of the recursive
         * resistance Rtilde, evaluated for the edge (s,h,a) at the current counts.
         *
         *                       C2                    T^{s'}(s,h,a)        1
         *      B^ER(s,h,a) = ---------- + C3 * sum_s' -------------- * ------------
         *                     T(s,h,a)                   T(s,h,a)       T(s',h+1)
         *                     \________/                 \____________________/
         *                    edge channel                  child channel (pooled counts)
         *
         * The edge channel is the resistance increment 1/N(e) of the traversed edge and covers
         * local reward/transition uncertainty. The child channel is the empirical expectation,
         * under phat_t, of the successor's own resistance 1/T(s',h+1) measured with the *global*
         * count pooled over every parent of s'. That pooling is the whole point: on a tree
         * T(s',h+1) equals T^{s'}(s,h,a) and each summand collapses to 1/T(s,h,a), while on a
         * graph it is damped by the sharing factor T^{s'}(s,h,a)/T(s',h+1) in (0,1] -- the bonus
         * shrinks exactly where transpositions have already contributed evidence.
         *
         * Deep Rtilde itself is never materialized: it does not vanish as T(s,h,a) -> infinity
         * (unexplored descendants keep it bounded away from zero), which is why the algorithm uses
         * this truncation, whose envelope 0 <= B^ER <= (C2 + C3*M)/T(s,h,a) is what the
         * convergence proof needs.
         *
         * Only called for initialized edges (visits >= 1); untried actions are selected first.
         */
        double er_bonus(const MCGraphEdgeStats &edge) const {
            if ((c2 == 0. && c3 == 0.) || edge.visits == 0) {
                return 0.;
            }

            double edge_visits = static_cast<double>(edge.visits);
            double bonus = c2 / edge_visits;

            if (c3 > 0.) {
                double child_channel = 0.;
                for (const auto &successor : edge.successors) {
                    // Every recorded successor sits on a backed-up path, hence has been visited at
                    // least once; the guard only keeps the ratio finite if that ever changes.
                    uint32_t pooled_visits = node_visits(successor.first);
                    double child_visits = static_cast<double>(pooled_visits > 0 ? pooled_visits : 1);
                    child_channel += (static_cast<double>(successor.second) / edge_visits) / child_visits;
                }
                bonus += c3 * child_channel;
            }

            return bonus;
        }

        uint32_t select_action(const S &state, const MCGraphNodeStats &node) {
            auto valid_actions = env->get_valid_actions(state);
            if (valid_actions.empty()) {
                throw std::runtime_error("No valid actions available in MCGraphSearchTree");
            }

            std::vector<uint32_t> candidates;
            double best_score = -std::numeric_limits<double>::infinity();
            bool saw_unvisited = false;

            for (uint32_t action : valid_actions) {
                auto it = node.edges.find(action);
                if (it == node.edges.end() || it->second.visits == 0) {
                    if (!saw_unvisited) {
                        saw_unvisited = true;
                        candidates.clear();
                    }
                    candidates.emplace_back(action);
                    continue;
                }

                if (saw_unvisited) {
                    continue;
                }

                double score = it->second.q_value +
                               c * std::pow(static_cast<double>(node.visits), 0.25) /
                               std::sqrt(static_cast<double>(it->second.visits)) +
                               er_bonus(it->second);
                if (score > best_score + 1e-12) {
                    best_score = score;
                    candidates.clear();
                    candidates.emplace_back(action);
                } else if (std::abs(score - best_score) <= 1e-12) {
                    candidates.emplace_back(action);
                }
            }

            if (candidates.empty()) {
                return random_utils->sample_uniform(valid_actions);
            }

            return random_utils->sample_uniform(candidates);
        }

        uint32_t select_greedy_action() {
            auto &node = ensure_node(current_state);
            auto valid_actions = env->get_valid_actions(current_state);
            if (valid_actions.empty()) {
                throw std::runtime_error("No valid actions available in MCGraphSearchTree");
            }
            std::vector<uint32_t> candidates;
            double best_score = -std::numeric_limits<double>::infinity();
            bool saw_visited = false;

            for (uint32_t action : valid_actions) {
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
                return random_utils->sample_uniform(valid_actions);
            }

            return random_utils->sample_uniform(candidates);
        }

        void run_rollout() {
            std::vector<MCGraphTransition> path;
            S sim_state = current_state;
            auto current_key = build_key(sim_state);
            bool use_selection = true;

            while (true) {
                uint32_t action;
                S next_state;
                uint32_t next_obs;
                double reward;
                bool done;

                if (use_selection) {
                    auto &current_node = ensure_node(current_key);
                    action = select_action(sim_state, current_node);
                    ensure_edge(current_key, action);
                    std::tie(next_state, next_obs, reward, done) = env->simulate(sim_state, action);
                } else {
                    std::tie(action, next_state, next_obs, reward, done) = env->random_simulate(sim_state);
                    ensure_edge(current_key, action);
                }

                auto next_key = build_key(next_state);
                auto next_entry = ensure_node_entry(next_key);
                bool next_seen_before = !next_entry.second;

                path.emplace_back(current_key, action, next_key, reward, done);

                if (done) {
                    break;
                }

                sim_state = next_state;
                current_key = next_key;
                use_selection = next_seen_before;
            }

            backup_path(path);
        }

        void backup_path(const std::vector<MCGraphTransition> &path) {
            if (path.empty()) {
                return;
            }

            auto &leaf = ensure_node(path.back().target);
            leaf.visits += 1;
            // if (path.back().terminal) {
            //     leaf.value = 0.;
            // }

            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                auto &target_node = ensure_node(it->target);
                auto &source_node = ensure_node(it->source);
                auto &edge = ensure_edge(it->source, it->action);

                edge.visits += 1;
                if (c3 > 0.) {
                    // T^{s'}_{s,h,a}: kept in lockstep with edge.visits so the child-channel
                    // weights of eq. (8) are a genuine empirical transition distribution.
                    edge.successors[it->target] += 1;
                }
                //double future_value = it->terminal ? 0. : target_node.value;
                double future_value = target_node.value;
                double backup_value = it->reward + effective_discount() * future_value;
                edge.q_value = ((edge.q_value * static_cast<double>(edge.visits - 1)) + backup_value) /
                               static_cast<double>(edge.visits);

                source_node.visits += 1;
                source_node.value = compute_node_value(source_node);
            }
        }

        double compute_node_value(const MCGraphNodeStats &node) const {
            if (node.visits == 0) {
                return node.value;
            }

            if (p == 0.) {
                double max_q = -std::numeric_limits<double>::infinity();
                bool saw_visited = false;
                for (const auto &entry : node.edges) {
                    const auto &edge = entry.second;
                    if (edge.visits == 0) {
                        continue;
                    }

                    if (!saw_visited || edge.q_value > max_q) {
                        max_q = edge.q_value;
                        saw_visited = true;
                    }
                }

                return saw_visited ? max_q : node.value;
            }

            double total = 0.;
            for (const auto &entry : node.edges) {
                const auto &edge = entry.second;
                if (edge.visits == 0) {
                    continue;
                }

                double weight = static_cast<double>(edge.visits) / static_cast<double>(node.visits);
                total += weight * std::pow(edge.q_value, p);
            }

            if (total <= 0.) {
                return 0.;
            }

            return std::pow(total, 1. / p);
        }
    };

}

#endif //C___MC_GRAPH_H
