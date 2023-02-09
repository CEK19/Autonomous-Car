from stable_baselines3.common.env_checker import check_env
from dynamic_obstacle_avoidance import DynamicObstacleAvoidance

env = DynamicObstacleAvoidance()
check_env(env)