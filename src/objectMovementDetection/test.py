from const import *
import numpy as np

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

print(lowerBound)
print(upperBound)