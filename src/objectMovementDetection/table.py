from const import *
from utils import *
import random
import numpy as np
import json

class RLAlgorithm:
    def __init__(self, rayCastingData, actions) -> None:
        # Converting n raycasting to signal in area , min raycast of each area
        self.signalPerAreaData = self.convertRayCastingDataToSignalPerArea(
            rayCastingData=rayCastingData)
        file = open(FILE.MODEL_SAVE, "r")
        RLInFile = file.read()
        if not RLInFile:
            self.Q = self._initQTable(actions=actions) 
        else:
            self.Q = json.loads(RLInFile)
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

        hashFromCenterOfLane = ""
        distanceFromCenterOfLane = abs(
            leftSideDistance - rightSideDistance) / 2
        for index, distance in enumerate(RLParam.DISTANCE_FROM_CENTER_OF_LANE):
            if index == len(RLParam.DISTANCE_FROM_CENTER_OF_LANE) - 1:
                hashFromCenterOfLane += RLParam.LEVEL_OF_LANE.MIDDLE
                break
            elif distanceFromCenterOfLane > distance:
                if leftSideDistance < rightSideDistance:
                    hashFromCenterOfLane += str(index +
                                                 int(RLParam.LEVEL_OF_LANE.MIDDLE) + 1)
                else:
                    hashFromCenterOfLane += str(index)
                break
        return hashFromRayCasting + hashFromCenterOfLane

    @staticmethod
    def getReward(currState, currActionIndex):
        finalReward = 0
        stateArr = [char for char in currState]
        lidarStates = stateArr[0:RLParam.AREA_RAY_CASTING_NUMBERS]
        centerState = stateArr[-1]

        # Obstacles block car
        for lidarState in lidarStates:
            if lidarState == RLParam.LEVEL_OF_RAY_CASTING.FAILED_DISTANCE:
                finalReward += RLParam.SCORE.OBSTACLE_TOUCH
            elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.DANGEROUS_DISTANCE:
                finalReward += RLParam.SCORE.DANGEROUS_ZONE_TOUCH

        # Car out of lane
        if centerState == RLParam.LEVEL_OF_LANE.MIDDLE:
            finalReward += RLParam.SCORE.STAY_AT_CENTER_OF_LANE
        elif centerState == RLParam.LEVEL_OF_LANE.RIGHT or centerState == RLParam.LEVEL_OF_LANE.LEFT:
            finalReward += RLParam.SCORE.STAY_AT_LEFT_OR_RIGHT_OF_LANE
        elif centerState == RLParam.LEVEL_OF_LANE.MOST_RIGHT or centerState == RLParam.LEVEL_OF_LANE.MOST_LEFT:
            finalReward += RLParam.SCORE.STAY_AT_MOSTLEFT_OR_MOSTRIGHT_OF_LANE

        # Prevent stop and go back action
        if RLParam.ACTIONS[currActionIndex] == PlayerParam.STOP:
            finalReward += RLParam.SCORE.STOP_ACTION
        elif RLParam.ACTIONS[currActionIndex] == PlayerParam.INC_ROTATION_VELO or RLParam.ACTIONS[currActionIndex] == PlayerParam.DESC_ROTATION_VELO:
            finalReward += RLParam.SCORE.TURN_LEFT_OR_RIGHT
            
        return finalReward

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
            # startTime = time.time()

            for actionCount in range(RLParam.MAX_EPISODE_STEPS):
                # print("state: ", state)
                actionIndex = self._epsilonGreedyPolicy(currState=state, currentEpsilon=epsilon)
                nextState, reward, done = env.updateStateByAction(actionIndex)
                totalReward += reward
                self.Q[state][actionIndex] = self.Q[state][actionIndex] + \
                    alpha * (reward + RLParam.GAMMA *
                             np.max(self.Q[nextState]) - self.Q[state][actionIndex])
                state = nextState

                if done or actionCount == RLParam.MAX_EPISODE_STEPS - 1: 
                    totalReward -=  (actionCount + 1) * 0.01 # 120s * 1 = 120
                    break
            comment = f"Episode {e + 1}, xPos={env.xPos} - yPos={env.yPos} : total reward in {actionCount} actions -> {totalReward}\n"
            print(comment, end="")
            
            progressFile = open(FILE.PROGRESS, "a")
            progressFile.write(comment)
            progressFile.close()
            if e % 20 == 0:
                print("--> start write to file")
                file = open(FILE.MODEL_SAVE, "w")
                file.write(json.dumps(self.Q))
                file.close()
                print("end write to file !!!")
            
            if(e % 50 == 0):
                print("backup Q table")
                file1 = open(FILE.MODEL_SAVE, "r")
                fileBackup = open("", "w")
                fileBackup.write(file1.read())
                fileBackup.close()
                file1.close()
                
                print("backup progress")
                file2 = open(FILE.PROGRESS, "r")
                fileBackup = open(FILE.PROGRESS_BACKUP, "w")
                fileBackup.write(file2.read())
                fileBackup.close()
                file2.close()
            
            env = env.reset()
