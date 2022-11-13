from const import *
from utils import *
import random
import numpy as np
import json
import time

# from main import getGlobalX

class RLAlgorithm:
    def __init__(self, rayCastingData, actions) -> None:
        self.signalPerAreaData = self.convertRayCastingDataToSignalPerArea(
            rayCastingData=rayCastingData)
        # file = open("rl-learning.txt", "r")
        # RLInFile = file.read()
        # if not RLInFile:
        #     self.Q = self._initQTable(actions=actions) 
        # else:
        #     self.Q = json.loads(RLInFile)
        #     print("Load completed")
        self.Q = self._initQTable(actions=actions) 
        self.actions = actions

    def _initQTable(self, actions):
        # https://www.geeksforgeeks.org/print-all-the-permutation-of-length-l-using-the-elements-of-an-array-iterative/
        rs = dict()
        numbersOfLevelRayCasting = len(RLParam.DISTANCE_OF_RAY_CASTING)
        listLevelOfRayCasting = list(range(numbersOfLevelRayCasting))        

        encodedAction = [0] * (len(self.signalPerAreaData))
        sizeEncodedAction = len(encodedAction)

        for i in range(pow(numbersOfLevelRayCasting, sizeEncodedAction)):
            k = i
            combinedString = ''
            for _ in range(sizeEncodedAction):
                combinedString += str(listLevelOfRayCasting[k %
                                      numbersOfLevelRayCasting])
                k //= numbersOfLevelRayCasting            
            
            for level in RLParam.LEVEL_OF_LANE.LIST_LEVEL_OF_LANE:    
                rs[combinedString + level] = [0] * len(RLParam.ACTIONS)            

        return rs

    @staticmethod
    def convertRayCastingDataToSignalPerArea(rayCastingData):
        # print("rayCastingData",rayCastingData)
        div = len(rayCastingData) // RLParam.AREA_RAY_CASTING_NUMBERS
        mod = len(rayCastingData) % RLParam.AREA_RAY_CASTING_NUMBERS

        tmpData = [0] * (RLParam.AREA_RAY_CASTING_NUMBERS +
                         (0 if mod == 0 else 1))
        
        tmpCount = 0
        for i in range(len(tmpData)):
            tmp = []
            for _ in range(div):
                tmp.append(rayCastingData[tmpCount])
                tmpCount += 1
                if (tmpCount == len(rayCastingData)):
                    break
            tmpData[i] = min(tmp)
        # print("tmpData",tmpData)
        return tmpData

    @staticmethod
    def hashFromDistanceToState(signalPerAreaData, leftSideDistance, rightSideDistance, angle):
        hashFromRayCasting = ""
        for signal in signalPerAreaData:
            for index, distanceRange in enumerate(RLParam.DISTANCE_OF_RAY_CASTING):
                if index == len(RLParam.DISTANCE_OF_RAY_CASTING) - 1:
                    hashFromRayCasting += RLParam.LEVEL_OF_RAY_CASTING.INFINITY
                    break
                elif signal < distanceRange:
                    hashFromRayCasting += str(index)
                    break

        hashFromCenterOfLane = ""
        if leftSideDistance < RLParam.LEVEL_OF_LANE.DISTANCE_MOST_LEFT:
            hashFromCenterOfLane += RLParam.LEVEL_OF_LANE.MOST_LEFT
        elif leftSideDistance < RLParam.LEVEL_OF_LANE.DISTANCE_LEFT:
            hashFromCenterOfLane += RLParam.LEVEL_OF_LANE.LEFT
        elif rightSideDistance < RLParam.LEVEL_OF_LANE.DISTANCE_MOST_RIGHT:
            hashFromCenterOfLane += RLParam.LEVEL_OF_LANE.MOST_RIGHT        
        elif rightSideDistance < RLParam.LEVEL_OF_LANE.DISTANCE_RIGHT:
            hashFromCenterOfLane += RLParam.LEVEL_OF_LANE.RIGHT
        else:
            hashFromCenterOfLane += RLParam.LEVEL_OF_LANE.MIDDLE
        
            
        hashFromAngle = ""
        if angle > RLParam.LEVEL_OF_ANGLE.NORMAL_LEFT_ANGLE and angle < RLParam.LEVEL_OF_ANGLE.NORMAL_RIGHT_ANGLE:
            hashFromAngle += RLParam.LEVEL_OF_ANGLE.FRONT
        elif angle < RLParam.LEVEL_OF_ANGLE.NORMAL_LEFT_ANGLE and angle > RLParam.LEVEL_OF_ANGLE.OVER_ROTATION_LEFT_ANGLE:
            hashFromAngle += RLParam.LEVEL_OF_ANGLE.NORMAL_LEFT                        
        elif angle > RLParam.LEVEL_OF_ANGLE.NORMAL_RIGHT_ANGLE and angle < RLParam.LEVEL_OF_ANGLE.OVER_ROTATION_RIGHT_ANGLE:
            hashFromAngle += RLParam.LEVEL_OF_ANGLE.NORMAL_RIGHT
        elif angle < RLParam.LEVEL_OF_ANGLE.OVER_ROTATION_LEFT_ANGLE:
            hashFromAngle += RLParam.LEVEL_OF_ANGLE.OVER_ROTATION_LEFT
        else:
            hashFromAngle += RLParam.LEVEL_OF_ANGLE.OVER_ROTATION_RIGHT            
        return hashFromRayCasting + hashFromCenterOfLane + hashFromAngle

    @staticmethod
    def getReward(currState, currActionIndex, currPlayer):
        finalReward = 0
        stateArr = [char for char in currState]
        lidarStates = stateArr[0:RLParam.AREA_RAY_CASTING_NUMBERS]
        centerState = stateArr[-1]

        # Obstacles block car
        for lidarState in lidarStates:
            if lidarState == 0:
                finalReward += -100
            elif lidarState == 1:
                finalReward += -10
            elif lidarState == 2:
                finalReward += -2

        # Car out of lane
        if centerState == RLParam.LEVEL_OF_LANE.MIDDLE:
            finalReward += 2
        elif centerState == RLParam.LEVEL_OF_LANE.RIGHT or centerState == RLParam.LEVEL_OF_LANE.LEFT:
            finalReward += -10
        elif centerState == RLParam.LEVEL_OF_LANE.MOST_RIGHT or centerState == RLParam.LEVEL_OF_LANE.MOST_LEFT:
            finalReward += -100

        # Prevent stop and go back action
        y_Ver = math.cos(currPlayer.currAngle)*currPlayer.currVelocity
        finalReward += -1*y_Ver*0.01
            
        return finalReward

    def _epsilonGreedyPolicy(self, currState):
        if np.random.uniform(0, 1) < RLParam.EPSILON:
            return random.choice(range(len(self.actions)))
        else:
            return np.argmax(self.Q[currState])

    def train(self, env):
        alphas = np.linspace(
            RLParam.MAX_ALPHA, RLParam.MIN_ALPHA, RLParam.N_EPISODES)

        for e in range(14800,RLParam.N_EPISODES):
            state = env.getCurrentState()
            totalReward = 0
            alpha = alphas[e]
            

            for actionCount in range(RLParam.MAX_EPISODE_STEPS):
                startTime = time.time()
                # print(" action:",actionCount,"episode:",e+1,"state:", state, "currentReward",totalReward,end="   ")
                actionIndex = self._epsilonGreedyPolicy(currState=state)
                nextState, reward, done = env.updateStateByAction(actionIndex)
                totalReward += reward
                self.Q[state][actionIndex] = self.Q[state][actionIndex] + \
                    alpha * (reward + RLParam.GAMMA *
                             np.max(self.Q[nextState]) - self.Q[state][actionIndex])
                state = nextState
                # print("state:", state)

                if done or actionCount == RLParam.MAX_EPISODE_STEPS - 1: 
                    # totalReward -=  (actionCount + 1) * 0.01 # 120s * 1 = 120
                    break
                # print("step time: ",time.time() - startTime)
            comment = f"Episode {e + 1}: total reward in {actionCount} -> {totalReward}\n"
            
            print(f"Episode {e + 1}: total reward in {actionCount} actions -> {totalReward}")

            if (e%1000 == 0):
                file = open("D:/RL/abc.txt", "w")
                file.write(json.dumps(self.Q))
                file.close()

                file2 = open("D:/RL/abc2.txt", "w")
                file2.write(json.dumps(self.Q))
                file2.close()
            
            progressFile = open("progress.txt", "a")
            progressFile.write(comment)
            
            env = env.reset()
