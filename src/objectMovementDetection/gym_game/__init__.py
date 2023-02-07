from gym.envs.registration import register
from const import *

register(
    id='AvoidDynamicObstacle-v0',
    entry_point='gym_game.envs:DynamicObstacleAvoidance',
    max_episode_steps=MAX_EPISODE,
)