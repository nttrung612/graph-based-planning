import os
import torch
import numpy as np
import re
from typing import Dict, Type, Any, Callable
from functools import lru_cache
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from mcts_forest.envs.gym_adapter import GymAdapter
from mcts_forest.core.gbopd import GBOPD
from mcts_forest.core.gbop import GBOP
from mcts_forest.envs.custom_envs import FactoredRiverSwimEnv, FourRoomsEnv, PassengerGridEnv, SysadminRingEnv

class Registry:
    def __init__(self):
        self.envs = {}
        self.solvers = {}

    def register_env(self, name, factory): self.envs[name.lower()] = factory
    def register_solver(self, name, factory): self.solvers[name.lower()] = factory
    
    def get_env(self, name, **kwargs):
        name_lower = name.lower()
        
        prob_match = re.search(r'_(\d+\.\d+)$', name_lower)
        if prob_match:
            prob = float(prob_match.group(1))
            base_name = name_lower[:prob_match.start()]
            if base_name in self.envs:
                if "frozenlake_slip" in base_name:
                    kwargs["success_rate"] = prob 
                return self.envs[base_name](**kwargs)

        river_match = re.match(r'riverswim_n(\d+)x(\d+)', name_lower)
        if river_match:
            kwargs["num_rivers"] = int(river_match.group(1))
            kwargs["num_locations"] = int(river_match.group(2))
            return self.envs["riverswim"](**kwargs)
            
        fourrooms_match = re.match(r'fourrooms_n(\d+)', name_lower)
        if fourrooms_match:
            kwargs["n"] = int(fourrooms_match.group(1))
            return self.envs["fourrooms"](**kwargs)
            
        sysadmin_match = re.match(r'sysadmin_n(\d+)', name_lower)
        if sysadmin_match:
            kwargs["num_computers"] = int(sysadmin_match.group(1))
            return self.envs["sysadmin"](**kwargs)
            
        return self.envs[name_lower](**kwargs)

    def get_solver(self, name, env, **kwargs): return self.solvers[name.lower()](env, **kwargs)

REGISTRY = Registry()

@lru_cache(maxsize=1024)
def get_map(size=4, seed=None):
    return generate_random_map(size=size, seed=seed)

REGISTRY.register_env("frozenlake", lambda **kwargs: GymAdapter("FrozenLake-v1", desc=get_map(4), is_slippery=False, **kwargs))
REGISTRY.register_env("frozenlake_slip", lambda **kwargs: GymAdapter("FrozenLake-v1", desc=get_map(4), is_slippery=True, **kwargs))
REGISTRY.register_env("frozenlake8x8", lambda **kwargs: GymAdapter("FrozenLake-v1", desc=get_map(8), is_slippery=False, **kwargs))
REGISTRY.register_env("frozenlake8x8_slip", lambda **kwargs: GymAdapter("FrozenLake-v1", desc=get_map(8), is_slippery=True, **kwargs))
REGISTRY.register_env("riverswim", lambda **kwargs: GymAdapter("FactoredRiverSwim-v0", **kwargs))
REGISTRY.register_env("fourrooms", lambda **kwargs: GymAdapter("FourRooms-v0", **kwargs))
REGISTRY.register_env("passenger_grid", lambda **kwargs: GymAdapter("PassengerGrid-v0", **kwargs))
REGISTRY.register_env("sysadmin", lambda **kwargs: GymAdapter("SysadminRing-v0", **kwargs))

REGISTRY.register_solver("gbopd", lambda env, **kwargs: GBOPD(env, **kwargs))
REGISTRY.register_solver("gbop", lambda env, **kwargs: GBOP(env, **kwargs))

