from const import *
from utils import *
import random
import numpy as np
import time


class RLAlgorithm:
    def __init__(self, rayCastingData, actions) -> None:
        # Converting n raycasting to signal in area , min raycast of each area
        self.signalPerAreaData = self.convertRayCastingDataToSignalPerArea(
            rayCastingData=rayCastingData)
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
        return tmpData

    @staticmethod
    def hashFromDistanceToState(signalPerAreaData, leftSideDistance, rightSideDistance):  # Tu
        hashFromRayCasting = ""
        for signal in signalPerAreaData:
            for index, distanceRange in enumerate(RLParam.DISTANCE_OF_RAY_CASTING):
                if index == len(RLParam.DISTANCE_OF_RAY_CASTING) - 1:
                    hashFromRayCasting += RLParam.LEVEL_OF_RAY_CASTING.INFINITY
                    break
                elif signal < distanceRange:
                    hashFromRayCasting += str(index)
                    break

        hashFromCeneterOfLane = ""
        distanceFromCenterOfLane = abs(
            leftSideDistance - rightSideDistance) / 2
        for index, distance in enumerate(RLParam.DISTANCE_FROM_CENTER_OF_LANE):
            if index == len(RLParam.DISTANCE_FROM_CENTER_OF_LANE) - 1:
                hashFromCeneterOfLane += RLParam.LEVEL_OF_LANE.MIDDLE
                break
            elif distanceFromCenterOfLane > distance:
                if leftSideDistance < rightSideDistance:
                    hashFromCeneterOfLane += str(index +
                                                 int(RLParam.LEVEL_OF_LANE.MIDDLE) + 1)
                else:
                    hashFromCeneterOfLane += str(index)
                break
        return hashFromRayCasting + hashFromCeneterOfLane

    @staticmethod
    def getReward(currState, currActionIndex):
        finalReward = 0
        stateArr = [char for char in currState]
        lidarStates = stateArr[0:RLParam.AREA_RAY_CASTING_NUMBERS]
        centerState = stateArr[-1]

        # Obstacles block car
        for lidarState in lidarStates:
            if lidarState == RLParam.LEVEL_OF_RAY_CASTING.FAILED_DISTANCE:
                print("1")
                return RLParam.SCORE.OBSTACLE_TOUCH
            elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.DANGEROUS_DISTANCE and RLParam.ACTIONS[currActionIndex] != PlayerParam.STOP:
                # print("2")
                finalReward += RLParam.SCORE.DANGEROUS_ZONE_CONTINUE_MOVING
            elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.DANGEROUS_DISTANCE and RLParam.ACTIONS[currActionIndex] == PlayerParam.STOP:
                # print("3")
                finalReward += RLParam.SCORE.DANGEROUS_ZONE_STOP

        # Car out of lane
        if centerState == RLParam.LEVEL_OF_LANE.MIDDLE:
            # print("4")
            finalReward += RLParam.SCORE.STAY_AT_CENTER_OF_LANE
        elif centerState == RLParam.LEVEL_OF_LANE.RIGHT or centerState == RLParam.LEVEL_OF_LANE.LEFT:
            # print("5")
            finalReward += RLParam.SCORE.STAY_AT_LEFT_OR_RIGHT_OF_LANE
        elif centerState == RLParam.LEVEL_OF_LANE.MOST_RIGHT or centerState == RLParam.LEVEL_OF_LANE.MOST_LEFT:
            # print("6")
            finalReward += RLParam.SCORE.STAY_AT_MOSTLEFT_OR_MOSTRIGHT_OF_LANE

        # Prevent stop and go back action
        if RLParam.ACTIONS[currActionIndex] == PlayerParam.STOP:
            finalReward += RLParam.SCORE.STOP_ACTION
        elif RLParam.ACTIONS[currActionIndex] == PlayerParam.DESC_FORWARD_VELO:
            finalReward += RLParam.SCORE.DESC_FORWARD_VELO_ACTION
            
        return finalReward

    def _epsilonGreedyPolicy(self, currState):
        if random.uniform(0, 1) < RLParam.EPSILON:
            return random.choice(range(len(self.actions)))
        else:
            return np.argmax(self.Q[currState])

    def train(self, env):
        alphas = np.linspace(
            RLParam.MAX_ALPHA, RLParam.MIN_ALPHA, RLParam.N_EPISODES)

        for e in range(RLParam.N_EPISODES):            
            state = env.getCurrentState()
            totalReward = 0
            alpha = alphas[e]
            startTime = time.time()

            for _ in range(RLParam.MAX_EPISODE_STEPS):
                actionIndex = self._epsilonGreedyPolicy(currState=state)
                print("action index: ", actionIndex)
                print("signal: ", self.signalPerAreaData)
                print("left, right: ", env.xPos, env.yPos)
                nextState, reward, done = env.updateStateByAction(actionIndex)
                totalReward += reward
                self.Q[state][actionIndex] = self.Q[state][actionIndex] + \
                    alpha * (reward + RLParam.GAMMA *
                             np.max(self.Q[nextState]) - self.Q[state][actionIndex])
                state = nextState
                if done:
                    endTime = time.time()
                    totalReward += 1500 - (endTime-startTime) * 10
                    break
            print(f"Episode {e + 1}: total reward -> {totalReward}")
            env.reset()
