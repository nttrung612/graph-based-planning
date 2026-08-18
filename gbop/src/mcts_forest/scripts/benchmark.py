import argparse
import time
import os
import numpy as np
import pandas as pd
import random
import sys
import gymnasium as gym
import ast
import itertools
import inspect
from typing import List, Dict, Any, Tuple
import concurrent.futures
import multiprocessing

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.rule import Rule
from rich import box

from mcts_forest.utils.registry import REGISTRY
from mcts_forest.utils.experiment import generate_experiment_name, clear_directory, parse_dict
from mcts_forest.utils.stats import bootstrap_stats

console = Console()

def parse_grid_item(value: Any) -> List[Any]:
    if isinstance(value, str):
        value = value.strip()
        if (value.startswith('(') and value.endswith(')')) or (value.startswith('[') and value.endswith(']')):
            try:
                res = ast.literal_eval(value)
                if isinstance(res, (list, tuple)):
                    return list(res)
                return [res]
            except:
                return [value]
    return [value]

def expand_params(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not params:
        return [{}]
    
    keys = list(params.keys())
    expanded_values = []
    for k in keys:
        v = params[k]
        if isinstance(v, (list, tuple)):
            expanded_values.append(list(v))
        else:
            expanded_values.append([v])
    
    combinations = list(itertools.product(*expanded_values))
    return [dict(zip(keys, combo)) for combo in combinations]

def get_solver_class(solver_name: str) -> Any:
    import mcts_forest.utils.registry as registry_mod
    mapping = {
        "gbop": "GBOP",
        "gbopd": "GBOPD",
    }
    class_name = mapping.get(solver_name.lower())
    if class_name:
        return getattr(registry_mod, class_name, None)
    return None

def filter_compatible(solver_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    solver_cls = get_solver_class(solver_name)
    if not solver_cls or not hasattr(solver_cls, '__init__'):
        return kwargs
    
    sig = inspect.signature(solver_cls.__init__)
    named_params = {p for p in sig.parameters if p not in ('self', 'env', 'kwargs', 'args')}
    universal_args = {
        'c', 'horizon', 'gamma', 'rollout_limit', 'simulation_limit', 
        'internal_reward_scale', 'internal_reward_offset', 'init_q', 
        'v_min', 'v_max', 'budget_strategy'
    }
    
    return {k: v for k, v in kwargs.items() if k in named_params or k in universal_args}

def run_episode(env_name, solver_name, sims, seed=None, episode_idx=0, solver_kwargs=None, **env_kwargs):
    solver_kwargs = solver_kwargs or {}
    env_kwargs_with_seed = env_kwargs.copy()
    if seed is not None: env_kwargs_with_seed["seed"] = seed
        
    env = REGISTRY.get_env(env_name, **env_kwargs_with_seed)
    
    reset_seed = (seed * 1000 + episode_idx) if seed is not None else None
    obs = env.reset(seed=reset_seed)

    solver = REGISTRY.get_solver(solver_name, env, simulation_limit=sims, **solver_kwargs)
    terminated, truncated = False, False
    total_reward, steps, search_times = 0, 0, []
    
    while not (terminated or truncated) and steps < 200:
        start_search = time.time()
        action, _ = solver.search(obs)
        search_times.append(time.time() - start_search)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
    success = (terminated and reward > 0)
    return total_reward, steps, success, np.mean(search_times) if search_times else 0.0

def worker_fn(task):
    try:
        np.random.seed(task["seed"] * 1000 + task["episode_idx"])
        random.seed(task["seed"] * 1000 + task["episode_idx"])
        
        reward, steps, success, avg_search_time = run_episode(
            task["env_name"], task["solver_name"], task["sims"],
            seed=task["seed"],
            episode_idx=task["episode_idx"],
            solver_kwargs=task["solver_kwargs"],
            **task["env_kwargs"]
        )
        
        return {
            "seed": task["seed"],
            "episode": task["episode_idx"] % task["ep_per_seed"],
            "total_reward": reward,
            "steps": steps,
            "success": int(success),
            "avg_search_time": avg_search_time
        }
    except Exception as e:
        return {
            "seed": task["seed"],
            "episode": task["episode_idx"] % task.get("ep_per_seed", 1),
            "total_reward": 0.0,
            "steps": 0,
            "success": 0,
            "avg_search_time": 0.0,
            "error": str(e)
        }

def main():
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        
    parser = argparse.ArgumentParser(description="GBOP Benchmarking Suite")
    parser.add_argument("--env", type=str, default="frozenlake", help="Environment ID")
    parser.add_argument("--solver", type=str, default="gbop", help="Solver ID")
    parser.add_argument("--sims", type=str, default="100", help="Simulations per move")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per seed")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    parser.add_argument("--log", action="store_true", help="Log output files")
    parser.add_argument("--solver_args", type=str, default="{}", help="Solver hyperparameters")
    parser.add_argument("--env_args", type=str, default="{}", help="Environment arguments")
    parser.add_argument("--table", action="store_true", help="Generate result_table.txt")
    
    args = parser.parse_args()
    
    envs = parse_grid_item(args.env)
    solvers = parse_grid_item(args.solver)
    sims_list = sorted([int(s) for s in parse_grid_item(args.sims)])
    base_solver_args = parse_dict(args.solver_args)
    base_env_args = parse_dict(args.env_args)
    
    all_experiments = []
    for env_name in envs:
        for solver_name in solvers:
            for sims in sims_list:
                expanded_solver_args = expand_params(base_solver_args)
                expanded_env_args = expand_params(base_env_args)
                
                unique_s_kwargs = []
                for s_kwargs in expanded_solver_args:
                    filtered = filter_compatible(solver_name, s_kwargs)
                    if filtered not in unique_s_kwargs:
                        unique_s_kwargs.append(filtered)
                
                for s_kwargs in unique_s_kwargs:
                    for e_kwargs in expanded_env_args:
                        all_experiments.append({
                            "env": env_name,
                            "solver": solver_name,
                            "sims": sims,
                            "solver_kwargs": s_kwargs,
                            "env_kwargs": e_kwargs
                        })

    if not all_experiments:
        console.print("[bold red]No experiments found.[/bold red]")
        return

    swept_keys = []
    if len(envs) > 1: swept_keys.append("env")
    if len(solvers) > 1: swept_keys.append("solver")
    if len(sims_list) > 1: swept_keys.append("sims")
    
    final_results_summary = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=10
    ) as progress:
        
        overall_task = progress.add_task("[bold cyan]Sweep Progress", total=len(all_experiments))
        
        for i, exp in enumerate(all_experiments):
            env_name, solver_name, sims = exp["env"], exp["solver"], exp["sims"]
            s_kwargs, e_kwargs = exp["solver_kwargs"], exp["env_kwargs"]
            
            label_parts = []
            for k in swept_keys:
                v = exp.get(k, s_kwargs.get(k, e_kwargs.get(k)))
                if v is not None:
                    label_parts.append(f"{k}={v}")
            current_label = " ".join(label_parts) or "Default"
            
            progress.update(overall_task, description=f"[bold cyan]Grid: {current_label}")
            
            temp_env = REGISTRY.get_env(env_name, **e_kwargs)
            temp_solver = REGISTRY.get_solver(solver_name, temp_env, simulation_limit=sims, **s_kwargs)
            
            if hasattr(temp_solver, "print_info"):
                temp_solver.print_info()

            output_path = None
            if args.log:
                experiment_name = generate_experiment_name(temp_env, temp_solver, seeds=args.seeds)
                output_path = os.path.join(args.output, experiment_name)
                clear_directory(output_path)
                os.makedirs(output_path, exist_ok=True)

            tasks = []
            for seed in range(args.seeds):
                for ep in range(args.episodes):
                    tasks.append({
                        "env_name": env_name, "solver_name": solver_name, "sims": sims,
                        "episode_idx": ep + seed * args.episodes, "seed": seed,
                        "ep_per_seed": args.episodes, "env_kwargs": e_kwargs, "solver_kwargs": s_kwargs
                    })

            exp_task = progress.add_task(f"  [dim]Episodes", total=len(tasks))
            results = []
            
            if args.parallel:
                num_workers = min(multiprocessing.cpu_count(), len(tasks))
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(worker_fn, t) for t in tasks]
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                        progress.advance(exp_task)
            else:
                for task in tasks:
                    results.append(worker_fn(task))
                    progress.advance(exp_task)
                    
            df = pd.DataFrame(results)
            progress.remove_task(exp_task)
            
            success_mean, success_bs_std = bootstrap_stats(df["success"].values)
            reward_mean, reward_bs_std = bootstrap_stats(df["total_reward"].values)
            avg_steps = df['steps'].mean()
            avg_time = df['avg_search_time'].mean()
            
            if args.log and output_path:
                df.to_csv(os.path.join(output_path, "results.csv"), index=False)
                
            final_results_summary.append({
                "Solver": solver_name,
                "Env": env_name,
                "Sims": sims,
                "Kwargs": str(s_kwargs) if s_kwargs else "-",
                "Success": f"{success_mean*100:.2f}% ± {success_bs_std*100:.2f}%",
                "Reward": f"{reward_mean:.4f} ± {reward_bs_std:.4f}",
                "Steps": f"{avg_steps:.2f}",
                "Time/Move": f"{avg_time:.4f}s"
            })
            
            progress.advance(overall_task)

    console.print("\n", Rule("Benchmarking Summary", style="bold cyan"))
    table = Table(box=box.MINIMAL_DOUBLE_HEAD, show_header=True, header_style="bold magenta")
    table.add_column("Solver")
    table.add_column("Env")
    table.add_column("Sims", justify="right")
    table.add_column("Solver Args", overflow="fold")
    table.add_column("Success Rate", justify="right")
    table.add_column("Mean Reward", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Time/Move", justify="right")

    for res in final_results_summary:
        table.add_row(
            res["Solver"], res["Env"], str(res["Sims"]), res["Kwargs"],
            res["Success"], res["Reward"], res["Steps"], res["Time/Move"]
        )
    console.print(table)
    
    if args.table:
        md_lines = ["| Solver | Env | Sims | Success | Reward |", "| --- | --- | --- | --- | --- |"]
        for res in final_results_summary:
            md_lines.append(f"| {res['Solver']} | {res['Env']} | {res['Sims']} | {res['Success']} | {res['Reward']} |")
        with open("result_table.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        console.print("[bold green]Table saved to result_table.txt[/bold green]")

if __name__ == "__main__":
    main()

