import gym
from gym import spaces
import numpy as np
from pygame_2d import PyGame2D
from const import *


class DynamicObstacleAvoidance(gym.Env):
    def __init__(self) -> None:
        super(DynamicObstacleAvoidance, self).__init__()
        self.pygame = PyGame2D()
        self.action_space = spaces.Discrete(ACTION_SPACE, )

        ratioLeft = (0, 1)
        alpha = (0, 2*math.pi)
        fwVelo = (0, PLAYER_SETTING.MAX_FORWARD_VELO)
        rVelo = (PLAYER_SETTING.MIN_ROTATION_VELO,
                 PLAYER_SETTING.MAX_ROTATION_VELO)

        lowerBoundLidar = np.full((PLAYER_SETTING.CASTED_RAYS,), 0, dtype=float)
        upperBoundLidar = np.full((PLAYER_SETTING.CASTED_RAYS,), INT_INFINITY, dtype=float)

        lowerBound = np.array([ratioLeft[0], alpha[0], fwVelo[0], rVelo[0]], dtype=float)
        lowerBound = np.concatenate((lowerBound, lowerBoundLidar))

        upperBound = np.array([ratioLeft[1], alpha[1], fwVelo[1], rVelo[1]], dtype=float)
        upperBound = np.concatenate((upperBound, upperBoundLidar))

        self.observation_space = spaces.Box(
            low=lowerBound,
            high=upperBound
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
