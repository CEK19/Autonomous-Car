from stable_baselines3 import PPO
import os
import time
import random
from datetime import date
from dynamic_obstacle_avoidance import DynamicObstacleAvoidance

today = date.today().strftime("%d-%m")

os.environ['KMP_DUPLICATE_LIB_OK']='True'



counter = 0
if os.path.exists(f"data/PPO-{today}"):
    counter = len(os.listdir(f"data/PPO-{today}"))

header_dir = f"data/PPO-{today}/{counter}"

models_dir = f"{header_dir}/model"
log_dir = f"{header_dir}/log"


env = DynamicObstacleAvoidance()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000

for i in range(100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}-{i}")

env.close()