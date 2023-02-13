import os
import time
import random
from dynamic_obstacle_avoidance import DynamicObstacleAvoidance
from sb3_contrib import RecurrentPPO

os.environ['KMP_DUPLICATE_LIB_OK']='True'
models_dir = f"models/PPO-{int(time.time())}"
log_dir = f"logs/PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = DynamicObstacleAvoidance()
env.reset()

model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 1000000
# for i in range(1, 100):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
#     model.save(f"{models_dir}/{TIMESTEPS*i}")

# env.close()


model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
model.save(f"{models_dir}")

env.close()