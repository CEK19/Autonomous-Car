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
                for angle in RLParam.LEVEL_OF_ANGLE.LIST_LEVEL_ANGLES:
                    for omega in RLParam.LEVEL_OF_ROTATION.LIST_LEVEL_OF_ROTATION:
                        rs[combinedString + level + angle + omega] = [0] * len(RLParam.ACTIONS)
                        for idx in range(len(RLParam.ACTIONS)):
                            rs[combinedString + level + angle + omega][idx] = np.random.uniform(-10, 11)
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
    def hashFromDistanceToState(signalPerAreaData, leftSideDistance, rightSideDistance, angle, RotationVelocity):
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
    
        hashFromOmega = ""
        if RotationVelocity >= RLParam.LEVEL_OF_ROTATION.MAX_RIGHT_ANGLE:
            hashFromOmega = RLParam.LEVEL_OF_ROTATION.MAX_RIGHT
        elif RotationVelocity > RLParam.LEVEL_OF_ROTATION.RIGHT_ANGLE:
            hashFromOmega = RLParam.LEVEL_OF_ROTATION.RIGHT
        elif RotationVelocity == RLParam.LEVEL_OF_ROTATION.CENTER_ANGLE:
            hashFromOmega = RLParam.LEVEL_OF_ROTATION.CENTER
        elif RotationVelocity > RLParam.LEVEL_OF_ROTATION.LEFT_ANGLE:
            hashFromOmega = RLParam.LEVEL_OF_ROTATION.LEFT
        elif RotationVelocity >= RLParam.LEVEL_OF_ROTATION.MAX_LEFT_ANGLE:
            hashFromOmega = RLParam.LEVEL_OF_ROTATION.MAX_LEFT
        
        return hashFromRayCasting + hashFromCenterOfLane + hashFromAngle + hashFromOmega

    @staticmethod
    def getReward(currState, previousInfo, currentInfo):
        # reward = f_t - f_(t-1) + g(t)
        def f(past, current):
            reward = 0
            
            preVelocity = past["velocity"]
            prevAngle = past["angle"]
            preYPos = past["yPos"]
            
            currVelocity = current["velocity"]            
            currAngle = current["angle"]
            currYPos = current["yPos"]
            
            if currYPos < 10:
                reward += RLParam.SCORE.FINISH_LINE
            elif currYPos < GameSettingParam.HEIGHT * (1 / 4):
                reward += 10000
            elif currYPos < GameSettingParam.HEIGHT * (2 / 4):
                reward += 1000
            elif currYPos < GameSettingParam.HEIGHT * (3 / 4):
                reward += 100
            else: 
                reward += 10
            # reward += (currYPos - preYPos)*RLParam.SCORE.INCREASE_Y
            # reward += ((-currVelocity*math.cos(currAngle)) - (-preVelocity*math.cos(prevAngle)))*RLParam.SCORE.INCREASE_SPEED_FORWARD
            return reward

        def g(state):
            finalReward = 0
            stateArr = [char for char in state]
            lidarStates = stateArr[0:RLParam.AREA_RAY_CASTING_NUMBERS]
            centerState = stateArr[-2]
            angleState = stateArr[-1]
            
            currForwardVelocity = currentInfo['velocity']
            currAngle = currentInfo['angle']

            # Obstacles block car
            for index, lidarState in enumerate(lidarStates):
                tempReward = 0
                if lidarState == RLParam.LEVEL_OF_RAY_CASTING.FAILED_DISTANCE:
                    tempReward = RLParam.SCORE.FAILED_DISTANCE_TOUCH
                elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.DANGEROUS_DISTANCE:
                    tempReward = RLParam.SCORE.DANGEROUS_ZONE_TOUCH
                elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.SAFETY_DISTANCE:
                    tempReward = RLParam.SCORE.SAFETY_ZONE_TOUCH
                
                if index == 0 or index == RLParam.AREA_RAY_CASTING_NUMBERS - 1:
                    finalReward += 0.3 * tempReward
                elif index == 1 or index == 4:
                    finalReward += 0.6 * tempReward
                else:
                    finalReward += tempReward
            
            # Angle of car
            if angleState == RLParam.LEVEL_OF_ANGLE.FRONT:
                finalReward += RLParam.SCORE.STAY_IN_FRONT
            elif angleState == RLParam.LEVEL_OF_ANGLE.NORMAL_LEFT or angleState == RLParam.LEVEL_OF_ANGLE.NORMAL_RIGHT:
                finalReward += RLParam.SCORE.STAY_IN_NORMAL_ANGLE
                    
            # Car out of lane
            if centerState == RLParam.LEVEL_OF_LANE.MIDDLE:
                finalReward += RLParam.SCORE.STAY_AT_CENTER_OF_LANE
            elif centerState == RLParam.LEVEL_OF_LANE.RIGHT or centerState == RLParam.LEVEL_OF_LANE.LEFT:
                finalReward += RLParam.SCORE.STAY_AT_LEFT_OR_RIGHT_OF_LANE
            elif centerState == RLParam.LEVEL_OF_LANE.MOST_RIGHT or centerState == RLParam.LEVEL_OF_LANE.MOST_LEFT:
                finalReward += RLParam.SCORE.STAY_AT_MOSTLEFT_OR_MOSTRIGHT_OF_LANE                        
            
            if (currForwardVelocity < 20):
                finalReward += RLParam.SCORE.STOPS_TO_ENJOY
            
            # if (currAngle <= math.pi/2 or currAngle >= math.pi + math.pi/2):
            #     finalReward += abs(math.pi - currAngle)*RLParam.SCORE.TURN_AROUND
                
            return finalReward

        totalReward = f(past=previousInfo, current=currentInfo) + g(state=currState)
        return totalReward

    def _epsilonGreedyPolicy(self, currState, currentEpsilon):
        if np.random.uniform(0, 1) < currentEpsilon:
            return random.choice(range(len(self.actions)))
        else:
            return np.argmax(self.Q[currState])

    def train(self, env):
        alphas = np.linspace(
            RLParam.MAX_ALPHA, RLParam.MIN_ALPHA, RLParam.N_EPISODES)
        epsilons = np.linspace(
            RLParam.MAX_EPSILON, RLParam.MIN_EPSILON, RLParam.N_EPISODES
        )
        for e in range(RLParam.N_EPISODES):
            state = env.getCurrentState()
            totalReward = 0
            alpha = alphas[e]
            epsilon = epsilons[e]

            for actionCount in range(RLParam.MAX_EPISODE_STEPS):
                startTime = time.time()
                # print(" action:",actionCount,"episode:",e+1,"state:", state, "currentReward",totalReward,end="   ")
                actionIndex = self._epsilonGreedyPolicy(currState=state, currentEpsilon=epsilon)
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

            if (e%1000 == 0 and e != 0) or (e == RLParam.N_EPISODES - 1):
                # file = open("D:/RL/abc.txt", "w")
                file = open("rl-learning.txt", "w")
                file.write(json.dumps(self.Q))
                file.close()

                # file2 = open("D:/RL/abc2.txt", "w")
                # file2.write(json.dumps(self.Q))
                # file2.close()
            
            progressFile = open("progress.txt", "a")
            progressFile.write(comment)
            
            env = env.reset()
