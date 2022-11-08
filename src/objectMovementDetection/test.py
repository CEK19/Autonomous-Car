# This file is just for testing
import   numpy as np
from const import *

epsilons = np.linspace(
    RLParam.MAX_EPSILON, RLParam.MIN_EPSILON, RLParam.N_EPISODES
)
print(epsilons)
k = np.random.uniform(0, 1)
print(k)