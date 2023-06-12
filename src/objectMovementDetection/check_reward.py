import time
from pygame_2d import PyGame2D
from utils import *
from copy import deepcopy
import numpy as np

game = PyGame2D()
ct = 0
rewardList = [0]*6
sleepTime = 0.05
pause = False
while True:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        sleepTime *= 2
    elif keys[pygame.K_d]:
        sleepTime /= 2
    # elif keys[pygame.K_s]:
    #     pause = True
    # elif keys[pygame.K_w]:
    #     print("bruh")
    #     pause = False

    preRobot = deepcopy(game.robot)
    for actionIndex in range(6):
        game.robot = deepcopy(preRobot)
        game.action(actionIndex)
        rewardList[actionIndex] = int(game.evaluate())
    bestAction = rewardList.index(max(rewardList))
    game.robot = deepcopy(preRobot)
    print(f"{rewardList[0]:>15} {rewardList[1]:>15} {rewardList[2]:>15} {rewardList[3]:>15} {rewardList[4]:>15} {rewardList[5]:>15} {rewardList.index(max(rewardList))}")
    game.action(bestAction)
    game.view()
    if game.is_done():
        print("Done")
        break
    
        
    time.sleep(sleepTime)
    