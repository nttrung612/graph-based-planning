# JAIR_cppv3

This repository contains a C++ implementation for empirical evaluation of
Monte-Carlo Tree Search (MCTS) algorithms on finite-horizon stochastic planning
benchmarks. The implementation is organized around a small set of environments,
standard MCTS baselines, and two graph-search variants proposed in this codebase.

The main source code is located in `c++/` and builds two executables:

- `dummy`: an MPI-based batch runner for repeated experiments and rollout-budget sweeps.
- `testmcgraph`: a single-episode diagnostic runner for the graph-search agents
  `gs_power_uct`, `gs_power_uct_f`, and `gs_power_uct_er`.

The current experimental suite includes five environments and the following algorithms:

- Environments: `frozen_lake`, `four_rooms`, `passenger_grid`, `factored_river_swim`,
  and `sysadmin_ring`.
- Algorithms: `max_uct`, `power_uct`, `ments`, `gs_power_uct`, `gs_power_uct_f`,
  and the effective-resistance variant `gs_power_uct_er`.

All experiments executed through `dummy` use discount factor `1.0` and replan at
every environment step. At each step, the planner receives the current rollout
budget, selects an action, executes the action in the true environment, and then
advances the corresponding tree or graph state.

## Repository Layout

```text
.
|-- c++/
|   |-- CMakeLists.txt
|   |-- main.cpp                 # MPI batch runner; builds target dummy
|   |-- testmcgraph.cpp          # diagnostic runner for MCGraph
|   |-- include/
|   |   |-- environment.h        # Environment<S> interface
|   |   |-- discrete_environment.h
|   |   |-- tree_search.h        # max_uct, power_uct, ments
|   |   |-- tree_search_nodes.h
|   |   |-- rollout_container.h
|   |   |-- graph_search.h       # graph key and graph-search mode definitions
|   |   |-- mc_graph.h           # gs_power_uct[_f] and gs_power_uct_er
|   |   `-- envs/
|   `-- src/
|       |-- discrete_environment.cpp
|       |-- random_utils.cpp
|       `-- envs/
|-- run_batch_common.sh          # shared helper functions for tuning/final runs
|-- run_tune.sh                  # parameter-grid tuning script
|-- run_final_config.sh          # selected final parameters by environment/algorithm
|-- run_final.sh                 # final experiment script
|-- run_result.txt               # current final-run output
`-- ENV_AGENT_GRAPH_RUNBOOK.md   # detailed internal runbook
```

## Algorithms

`max_uct`

- A tree-search baseline based on UCT, using a UCB-style score for action selection.
- Command-line parameter: `alpha`.
- In this implementation, the V-node backup uses the maximum empirical Q-value
  among visited actions.
- The relevant foundational reference is Kocsis and Szepesvari (2006),
  "Bandit Based Monte-Carlo Planning".

`power_uct`

- A tree-search method with a polynomial exploration bonus and a power-mean backup.
- Command-line parameters: `alpha p`.
- When `p = 0`, the implementation uses a max backup. When `p > 0`, the node value
  is computed as a visit-weighted power mean over action Q-values.
- The relevant references are Dam et al. (2020) for Power-UCT/generalized mean
  estimation, and Dam, Maillard, and Kaufmann (2024) for the stochastic-MCTS
  power-mean analysis.

`ments`

- Maximum Entropy for Tree Search (MENTS), which evaluates nodes through soft
  values and uses Boltzmann action likelihoods with additional epsilon exploration.
- Command-line parameters: `tau epsilon`.
- The relevant reference is Xiao, Huang, Mei, Schuurmans, and Muller (2019),
  "Maximum Entropy Monte-Carlo Planning".

`gs_power_uct`

- A proposed graph-search variant implemented in `c++/include/mc_graph.h`.
- Command-line parameters: `c p`.
- This variant uses `GraphSearchMode::Depth`: the graph state key includes `time`,
  so the same physical state reached at different depths is represented by
  distinct graph nodes.
- Action selection uses `Q + c * N^0.25 / sqrt(n)`. Backups use a max operator
  when `p = 0`, and a power mean when `p > 0`.

`gs_power_uct_f`

- A second proposed graph-search variant implemented by the same MCGraph code.
- Command-line parameters: `c p`.
- This variant uses `GraphSearchMode::Full`: the graph key sets `time = 0`, which
  merges identical physical states across depths.
- The purpose of this variant is to evaluate whether cross-depth graph sharing
  improves planning performance relative to depth-specific graph nodes.

`gs_power_uct_er`

- GS-Power-UCT-ER: `gs_power_uct` plus the effective-resistance exploration bonus,
  implemented by the same MCGraph code in `c++/include/mc_graph.h`.
- Command-line parameters: `c p c2 c3`. With `c2 = c3 = 0` the algorithm reduces
  exactly to `gs_power_uct`.
- Action selection adds the one-step truncation of the recursive resistance to the
  usual score:

  ```
                       C2                    T(s,h,a,s')        1
      B_ER(s,h,a) = ---------- + C3 * sum_s' ------------ * ------------
                     T(s,h,a)                  T(s,h,a)      T(s',h+1)
  ```

  where `T(s,h,a,s')` counts realized transitions along the edge and `T(s',h+1)` is
  the successor's visit count pooled over *all* of its parents. The first term is
  the edge channel (local reward/transition uncertainty); the second is the child
  channel, which is damped by the sharing factor `T(s,h,a,s')/T(s',h+1)` exactly
  where transpositions have already contributed evidence.
- The bonus is confined to action selection. Backups, stored estimates and the
  recommended action (`argmax Q`) are unperturbed, so the algorithm targets the same
  fixed point as `gs_power_uct` and only changes which actions get sampled.
- Suggested coefficients: `C2 = C3`, of a magnitude comparable to `c`, swept over
  `{0, 0.1, 0.5, 1, 2}`.
- **Depth-augmented graph only — there is no `_f` counterpart.** The resistance the
  bonus truncates is defined by a topological recursion over a layered DAG, which
  `GraphSearchMode::Depth` guarantees because every edge advances `time` by one.
  Cross-depth merging can close cycles, where that recursion has no fixed point;
  passing ER coefficients together with `GraphSearchMode::Full` therefore throws in
  `MCGraphSearchTree`'s constructor rather than silently producing a number.

## Environments

`frozen_lake`

- Files: `c++/include/envs/frozen_lake.h`, `c++/src/envs/frozen_lake.cpp`.
- An 8x8 grid world with four actions: left, down, right, and up.
- The batch runner instantiates `FrozenLake(true, false)`: slippery dynamics are
  enabled and the rollout heuristic is disabled.
- The reward is `1` upon reaching the goal and `0` otherwise. Episodes terminate
  at the goal, at holes, or at the time limit of `200`.

`four_rooms`

- Files: `c++/include/envs/four_rooms.h`, `c++/src/envs/four_rooms.cpp`.
- Default constants are `FOUR_ROOMS_N = 5`, grid size `11x11`, and time limit `50`.
- Each reset samples a start state, a goal state, and four door cells. Actions are
  stochastic, with left/intended/right slip probabilities `0.25/0.5/0.25`.
- The terminal goal reward is `1 - 0.9 * step_count / 50`.
- `main.cpp` uses `run_randomized_initial_experiment()` so that `env` and
  `env_search` share the same randomized initial configuration.

`passenger_grid`

- Files: `c++/include/envs/passenger_grid.h`, `c++/src/envs/passenger_grid.cpp`.
- A 7x6 grid with start `(0,0)`, goal `(6,0)`, and three passengers at `(1,2)`,
  `(0,5)`, and `(6,4)`.
- The action set is left, down, right, and up. Slippery dynamics are enabled in
  the batch runner.
- The time limit is a command-line environment parameter; the scripts use `70`.
- Rewards are issued only at the goal and depend on the number of collected
  passengers: `0, 1, 3, 7`.

`factored_river_swim`

- Files: `c++/include/envs/factored_river_swim.h`,
  `c++/src/envs/factored_river_swim.cpp`.
- The default instance has 4 independent rivers, 8 locations per river, and time
  limit `35`.
- An action is a 4-bit mask, giving `2^4 = 16` actions.
- The reward includes a small reward for resting at the first location, a reward
  for swimming at the goal location, and a bonus when all rivers are at their
  goal and all action bits select swim. The total reward is normalized by
  `2 * num_rivers`.

`sysadmin_ring`

- Files: `c++/include/envs/sysadmin_ring.h`, `c++/src/envs/sysadmin_ring.cpp`.
- The default instance contains 20 computers arranged in a ring and uses time
  limit `50`.
- Actions `0..19` reboot individual machines; action `20` is idle.
- The state is represented by `alive_mask` and `time`. The per-step reward is the
  fraction of computers running after the transition.

## Main Execution Flow

`dummy` (`c++/main.cpp`) is the main experimental runner:

1. Parse the command line: `N_EXPERIMENTS`, algorithm, environment, environment
   arguments, and algorithm parameters.
2. Use MPI rank 0 as the coordinator and ranks `1..np-1` as workers.
3. For each rollout budget in `rollout_map`, rank 0 broadcasts the budget to the
   workers.
4. Each worker evaluates a subset of seeds from `N_EXPERIMENTS`.
5. Each experiment constructs `env` and `env_search`, seeds them consistently,
   replans at each step, and accumulates reward with discount factor `1.0`.
6. Rank 0 gathers rewards, computes mean and sample standard deviation, and
   prints a `Run Summary` block.

The current rollout budgets in `main.cpp` are identical for all environments:

```text
16, 32, 64, 128, 256, 512, 1024
```

`testmcgraph` (`c++/testmcgraph.cpp`) is a diagnostic runner:

- It accepts only `gs_power_uct` and `gs_power_uct_f`.
- It runs one episode and prints the current state, action Q-values and visit
  counts, selected action, effective action under slippery dynamics, next state,
  reward, and terminal flag.
- It is intended for inspecting graph keys, backups, and action-selection
  behavior in MCGraph.

## Ubuntu / WSL Setup

The experiments require a CPU, a C++ compiler, CMake, and MPI. A GPU is not
required.

On Ubuntu or WSL Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake openmpi-bin libopenmpi-dev
```

From Windows PowerShell, first enter WSL:

```bash
wsl
cd /mnt/d/trandinhtung/research/JAIR_cppv3
```

Check the number of available logical CPUs before selecting the MPI process
count:

```bash
nproc
```

In MPI runs, rank 0 is the coordinator. Therefore, `-np 8` creates 7 actual
workers. As a practical rule, choose `-np` no larger than the number of logical
CPUs unless intentional oversubscription is desired. If OpenMPI reports that
there are not enough slots, reduce `-np` or use `--oversubscribe` only when CPU
contention is acceptable.

## Build

From the repository root:

```bash
cd c++
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
cd ..
```

After a successful build, the expected executables are:

```text
c++/build/dummy
c++/build/testmcgraph
```

## Single Environment/Algorithm Runs

General syntax for `dummy`:

```bash
mpirun -np <NP> ./c++/build/dummy N_EXPERIMENTS max_uct ENV ALPHA
mpirun -np <NP> ./c++/build/dummy N_EXPERIMENTS power_uct ENV ALPHA P
mpirun -np <NP> ./c++/build/dummy N_EXPERIMENTS ments ENV TAU EPSILON
mpirun -np <NP> ./c++/build/dummy N_EXPERIMENTS gs_power_uct ENV C P
mpirun -np <NP> ./c++/build/dummy N_EXPERIMENTS gs_power_uct_f ENV C P
mpirun -np <NP> ./c++/build/dummy N_EXPERIMENTS gs_power_uct_er ENV C P C2 C3
```

For `passenger_grid`, add `TIME_LIMIT` immediately after the environment name:

```bash
mpirun -np <NP> ./c++/build/dummy N_EXPERIMENTS gs_power_uct passenger_grid TIME_LIMIT C P
```

Example commands:

```bash
mpirun -np 4 ./c++/build/dummy 100 max_uct frozen_lake 1.5
mpirun -np 4 ./c++/build/dummy 100 power_uct four_rooms 1.0 3.0
mpirun -np 4 ./c++/build/dummy 100 ments sysadmin_ring 0.25 0.5
mpirun -np 4 ./c++/build/dummy 100 gs_power_uct passenger_grid 70 0.5 0.0
mpirun -np 4 ./c++/build/dummy 100 gs_power_uct_f factored_river_swim 0.5 2.5
mpirun -np 4 ./c++/build/dummy 100 gs_power_uct_er four_rooms 0.5 0.0 0.5 0.5
mpirun -np 4 ./c++/build/dummy 100 gs_power_uct_er passenger_grid 70 0.5 0.0 0.5 0.5
```

Each run ends with a summary block of the following form:

```text
===== Run Summary =====
n_experiments: ...
algorithm: ...
environment: ...
mean_performance_by_rollouts:
rollouts=16, mean_reward=..., std_reward=..., 2std=...
...
=======================
```

## MCGraph Debugging

`testmcgraph` does not use MPI and is limited to the MCGraph agents. The two ER
coefficients are supplied exactly for the `_er` algorithms.

General syntax:

```bash
./c++/build/testmcgraph gs_power_uct ENV N_ROLLOUTS C P [SEED]
./c++/build/testmcgraph gs_power_uct_f ENV N_ROLLOUTS C P [SEED]
./c++/build/testmcgraph gs_power_uct_er ENV N_ROLLOUTS C P C2 C3 [SEED]
```

For `passenger_grid`, add `TIME_LIMIT`:

```bash
./c++/build/testmcgraph gs_power_uct passenger_grid TIME_LIMIT N_ROLLOUTS C P [SEED]
./c++/build/testmcgraph gs_power_uct_er passenger_grid TIME_LIMIT N_ROLLOUTS C P C2 C3 [SEED]
```

Example commands:

```bash
./c++/build/testmcgraph gs_power_uct frozen_lake 64 0.5 0.0 0
./c++/build/testmcgraph gs_power_uct_f four_rooms 128 0.5 4.0 1
./c++/build/testmcgraph gs_power_uct passenger_grid 70 64 0.5 0.0 0
./c++/build/testmcgraph gs_power_uct_er four_rooms 64 0.5 0.0 0.5 0.5 0
```

## Parameter Tuning

`run_tune.sh` performs a parameter-grid search over all configured
environment/algorithm pairs and writes summary blocks to `tune_result.txt`.
The script uses the MPI command defined in `run_batch_common.sh`:

```bash
MPI_CMD=(mpirun -np 8 "$DUMMY_BINARY")
```

Before running the tuning sweep, check the available CPU count:

```bash
nproc
```

Then adjust `-np 8` in `run_batch_common.sh` if necessary. Practical choices are:

- 4 logical CPUs: use `-np 4` or `-np 3`.
- 8 logical CPUs: `-np 8` is reasonable and creates 7 workers.
- 16 logical CPUs: `-np 12` or `-np 16` may be appropriate if memory and thermal
  constraints are acceptable.

Run the tuning sweep with:

```bash
bash run_tune.sh
```

Default settings in `run_tune.sh`:

- `N_EXPERIMENTS=300`.
- Environments: `frozen_lake`, `four_rooms`, `passenger_grid`,
  `factored_river_swim`, and `sysadmin_ring`.
- Algorithms: `max_uct`, `power_uct`, `gs_power_uct`, `gs_power_uct_f`,
  `gs_power_uct_er`, and `ments`.
- For `gs_power_uct_er` the `(c, p)` grid is deliberately narrow (`er_c_values`,
  `er_p_values`) because the base exploration parameters are already covered by the
  `gs_power_uct` sweep; what is swept is the coefficient grid `er_coefficient_pairs`,
  tied as `C2 = C3` over `{0.1, 0.5, 1, 2}`. Add pairs such as `"0.5 0.0"` or
  `"0.0 0.5"` there to measure the edge and child channels separately.
- `passenger_grid` uses `TIME_LIMIT=70` through `build_env_args()`.

After inspecting `tune_result.txt`, select the best parameters for each
`(environment, algorithm)` pair and enter them in `run_final_config.sh`.

## Final Experiments

`run_final.sh` executes the fixed parameter settings in `run_final_config.sh` and
writes summary blocks to `run_result.txt`.

Check or edit the MPI `-np` value in `run_batch_common.sh`, then run:

```bash
bash run_final.sh
```

Default settings in `run_final.sh`:

- `N_EXPERIMENTS=1000`.
- The script evaluates all 5 environments and every algorithm listed in its
  `algorithms` array. `gs_power_uct_er` is listed but commented out: its entries in
  `run_final_config.sh` are placeholders (the tuned `gs_power_uct` `(C, P)` with
  `C2 = C3 = 0.5`) and should be replaced from `tune_result.txt` before it is
  enabled.
- Each case is evaluated over the rollout budgets `16..2048` defined in
  `main.cpp`.
- Each invocation resets `run_result.txt`.

## Modifying or Adding Environments

The current compile-time environment sizes are defined in the corresponding
headers:

- `FOUR_ROOMS_N`, `FOUR_ROOMS_TIME_LIMIT`: `c++/include/envs/four_rooms.h`.
- `FACTORED_RIVER_SWIM_NUM_RIVERS`, `FACTORED_RIVER_SWIM_NUM_LOCATIONS`,
  `FACTORED_RIVER_SWIM_TIME_LIMIT`: `c++/include/envs/factored_river_swim.h`.
- `SYSADMIN_RING_NUM_COMPUTERS`, `SYSADMIN_RING_TIME_LIMIT`:
  `c++/include/envs/sysadmin_ring.h`.
- `PassengerGrid` receives its time limit through the command line or scripts;
  it is not a compile-time constant.

To add a new environment, add the header and source files, include the source in
`COMMON_ENV_SOURCES` in `c++/CMakeLists.txt`, add construction and CLI branches
in `main.cpp`, add rollout budgets to `rollout_map`, and, if MCGraph support is
required, implement `GraphStateTraits<S>` in `c++/include/graph_search.h`.

## References

- Kocsis, L. and Szepesvari, C. (2006). "Bandit Based Monte-Carlo Planning."
  ECML 2006, LNCS 4212, pp. 282-293. DOI: https://doi.org/10.1007/11871842_29
- Xiao, C., Huang, R., Mei, J., Schuurmans, D. and Muller, M. (2019).
  "Maximum Entropy Monte-Carlo Planning." NeurIPS 2019.
  https://papers.neurips.cc/paper/9148-maximum-entropy-monte-carlo-planning
- Dam, T., Klink, P., D'Eramo, C., Peters, J. and Pajarinen, J. (2020).
  "Generalized Mean Estimation in Monte-Carlo Tree Search." IJCAI 2020,
  pp. 2397-2404. https://www.ijcai.org/proceedings/2020/332
- Dam, T., Maillard, O.-A. and Kaufmann, E. (2024). "Power Mean Estimation in
  Stochastic Monte-Carlo Tree Search." UAI 2024, PMLR 244:894-918.
  https://proceedings.mlr.press/v244/dam24a.html
