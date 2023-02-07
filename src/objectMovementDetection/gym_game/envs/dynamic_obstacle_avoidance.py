import gym
from gym import spaces
import numpy as np
from gym_game.envs.pygame_2d import PyGame2D
from const import *


class DynamicObstacleAvoidance(gym.Env):
    def __init__(self) -> None:
        self.pygame = PyGame2D()
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.observation_space = spaces.Box(
            # TODO: FILL IN VALUE HERE
            np.array([]),
            np.array([])
        )

    def reset(self):
        del self.pygame
        self.pygame = PyGame2D()
        obs = self.pygame.observe()
        return obs

    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.pygame.view()
