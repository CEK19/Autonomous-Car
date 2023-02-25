"""
Env 2D
@author: Minh The Tus
"""

from const import *

class Env:
    '''
        maps: 2d array
    '''
    def __init__(self, maps):
        self.width = len(maps[0])  
        self.height = len(maps)
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.obs_map(maps)

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self, maps):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """
        obs = set()
        
        for rowIndex in range(self.height):
            for colIndex in range(self.width):
                if maps[rowIndex][colIndex] == D_STAR.ENV.HAS_OBS:
                    obs.add((rowIndex, colIndex))
                    

        return obs
