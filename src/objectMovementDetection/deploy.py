# import os
# import random
# import time

# from sb3_contrib import RecurrentPPO

# from dynamic_obstacle_avoidance import DynamicObstacleAvoidance

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# models_dir = f"models/PPO-{int(time.time())}"

# env = DynamicObstacleAvoidance()

# TIMESTEPS = 1000000


# episodes = 10000
# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         env.render()
#         print(rewards)