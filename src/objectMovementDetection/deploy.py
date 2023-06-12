import os
import time
import random
from dynamic_obstacle_avoidance import DynamicObstacleAvoidance
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from datetime import date
import sys

days = "data/PPO-13-02"
modelGroup = os.listdir(days)
modelGroup.sort(key = len)

modleCounter = os.listdir(f"{days}/{modelGroup[-1]}")
modleCounter.sort(key=len)

env = DynamicObstacleAvoidance()
env.reset()



for eachModel in modleCounter[-5:]:
    model_path = f"{days}/{modelGroup[-1]}/{eachModel}"
    print(f"{days}/{modelGroup[-1]}/{eachModel}")
    model = PPO.load(model_path, env=env)

    episodes = 5

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ct = 0
        while not done:
            ct+=1
            if ct > 500:
                break
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            # print(rewards)
            # print(obs, rewards, done, info)