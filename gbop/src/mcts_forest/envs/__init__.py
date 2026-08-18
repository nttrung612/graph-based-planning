from gymnasium.envs.registration import register

register(
    id='FactoredRiverSwim-v0',
    entry_point='mcts_forest.envs.custom_envs:FactoredRiverSwimEnv',
)

register(
    id='FourRooms-v0',
    entry_point='mcts_forest.envs.custom_envs:FourRoomsEnv',
)

register(
    id='PassengerGrid-v0',
    entry_point='mcts_forest.envs.custom_envs:PassengerGridEnv',
)

register(
    id='SysadminRing-v0',
    entry_point='mcts_forest.envs.custom_envs:SysadminRingEnv',
)
