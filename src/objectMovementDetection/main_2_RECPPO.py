import os
import time
import random
from stable_baselines3 import PPO
from dynamic_obstacle_avoidance import DynamicObstacleAvoidance
from sb3_contrib import RecurrentPPO
from datetime import date
import shutil

today = date.today().strftime("%d-%m")

os.environ['KMP_DUPLICATE_LIB_OK']='True'

shutil.rmtree("./LastRun")
os.mkdir("./LastRun")

counter = 0
if os.path.exists(f"data/PPO-{today}"):
    counter = len(os.listdir(f"data/PPO-{today}"))

header_dir = f"data/PPO-{today}/{counter}"

models_dir = f"{header_dir}/model"
log_dir = f"{header_dir}/log"


env = DynamicObstacleAvoidance()
env.reset()

model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=log_dir)
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
# model = RecurrentPPO.load("./data/PPO-12-02/15/model-14", env=env)

TIMESTEPS = 10000

for i in range(0,100000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}-{i}")

env.close()

# tensorboard --logdir=D:/github-cloneZone/Autonomous-Car/src/objectMovementDetection/data/PPO-12-02/1/log/PPO_0  --port=6006