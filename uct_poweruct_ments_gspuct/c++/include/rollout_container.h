#ifndef C___ROLLOUT_CONTAINER_H
#define C___ROLLOUT_CONTAINER_H

#include <tree_search_nodes.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <map>
#include <memory>
#include <functional>


namespace poweruct {

    template<typename S, typename I>
    class QNode;

    template<typename S, typename I>
    class VNode;


    template<typename S, typename I>
    class RolloutContainer {

    public:
        RolloutContainer(uint32_t na, double discount_factor,
                         std::function<std::shared_ptr<I>(S, uint32_t, bool)> node_constructor)
                : na(na), discount_factor(discount_factor), total_reward(0.), expanded(false),
                  node_constructor(node_constructor) {}

        ~RolloutContainer() = default;

        double get_total_reward() {
            return total_reward;
        }

        uint32_t get_total_visits() {
            return ((uint32_t) trajectories.size());
        }

        void store_rollout(std::unique_ptr<std::vector<std::tuple<uint32_t, double, S, uint32_t> > > trajectory) {
            if (expanded) {
                throw std::runtime_error("Once a RolloutContainer is expanded, no more trajectories can be added!");
            }

            total_reward += compute_trajectory_reward(trajectory);
            trajectories.emplace_back(std::move(trajectory));
        }

        std::map<uint32_t, std::shared_ptr<QNode<S, I> > > expand() {
            if (expanded) {
                throw std::runtime_error("RolloutContainer is already expanded!");
            }

            std::map<uint32_t, std::shared_ptr<QNode<S, I> > > res;
            for (auto &&trajectory : trajectories) {
                uint32_t n_steps = trajectory->size();
                if (n_steps > 0) {
                    auto step = trajectory->at(0);
                    auto action = std::get<0>(step);
                    auto reward = std::get<1>(step);
                    auto state = std::get<2>(step);
                    auto observation = std::get<3>(step);

                    auto it = res.find(action);

                    // If no QNode with the given action exists, we create a new one
                    std::shared_ptr<QNode<S, I>> q_node;
                    if (it == res.end()) {
                        q_node = std::make_shared<QNode<S, I>>(action, discount_factor);
                        res[action] = q_node;
                    } else {
                        q_node = it->second;
                    }

                    // Now we add the trajectory to the node
                    auto v_node = q_node->get_children(observation);
                    // Again if no v_node exists, we create one
                    if (!v_node) {
                        v_node = node_constructor(state, observation, n_steps == 1);
                        q_node->set_children(observation, v_node);
                        v_node->set_parent(q_node);
                    }

                    // Now we can create the remainder of the rollout and the immediate reward and update the q_node value
                    trajectory->erase(trajectory->begin());
                    v_node->store_rollout(std::move(trajectory));
                    q_node->store_reward(reward);
                    q_node->update(observation);
                }
            }

            expanded = true;
            total_reward = 0.;
            trajectories.clear();

            return res;
        }

    private:
        uint32_t na;
        double discount_factor;
        bool expanded;
        std::vector<std::unique_ptr<std::vector<std::tuple<uint32_t, double, S, uint32_t> > > > trajectories;
        std::function<std::shared_ptr<I>(S, uint32_t, bool)> node_constructor;
        double total_reward;

        double compute_trajectory_reward(
                const std::unique_ptr<std::vector<std::tuple<uint32_t, double, S, uint32_t> >> &trajectory) {
            double res = 0.;
            double disc = 1.;
            uint32_t size = trajectory->size();
            for (uint32_t i = 0; i < size; i++) {
                res += disc * std::get<1>(trajectory->at(i));
                disc *= discount_factor;

            }

            return res;
        }

    };

}


#endif //C___ROLLOUT_CONTAINER_H
