from const import *
import random
import numpy as np
import time


class RLAlgorithm:
    def __init__(self, rayCastingData, actions) -> None:
        # Converting n raycasting to signal in area , min raycast of each area
        self.signalPerAreaData = self._initSignalPerArea(rayCastingData=rayCastingData)
        self.leftSideDistance = 16
        self.rightSideDistance = 19
                
        self.Q = self._initQTable(actions=actions)
        self.actions = actions

    def _initSignalPerArea(self, rayCastingData):
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

    def _initQTable(self, actions):
        pass
    
    def _hashFromDistanceToState(self): # Tu
        hashFromRayCasting = ""
        for signal in self.signalPerAreaData:
            for index, distanceRange in enumerate(RLParam.DISTANCE_OF_RAY_CASTING):
                if index == len(RLParam.DISTANCE_OF_RAY_CASTING) - 1:
                    hashFromRayCasting += RLParam.LEVEL_OF_RAY_CASTING.INFINITY
                    break
                elif signal < distanceRange:
                    hashFromRayCasting += str(index)
                    break
        
        hashFromCeneterOfLane = ""
        distanceFromCenterOfLane = abs(self.leftSideDistance - self.rightSideDistance) / 2
        for index, distance in enumerate(RLParam.DISTANCE_FROM_CENTER_OF_LANE):
            if index == len(RLParam.DISTANCE_FROM_CENTER_OF_LANE) - 1:
                hashFromCeneterOfLane += RLParam.LEVEL_OF_LANE.MIDDLE
                break
            elif distanceFromCenterOfLane > distance:
                if self.leftSideDistance < self.rightSideDistance:
                    hashFromCeneterOfLane += str(index + int(RLParam.LEVEL_OF_LANE.MIDDLE) + 1)
                else:
                    hashFromCeneterOfLane += str(index)
                break
        return hashFromRayCasting + hashFromCeneterOfLane
    
    def getReward(self, currState, currActionIndex): 
        finalReward = 0
        stateArr = [int(char) for char in currState]
        lidarStates = stateArr[0:RLParam.AREA_RAY_CASTING_NUMBERS]
        centerState = stateArr[-1]
        
        # Obstacles block car
        for lidarState in lidarStates:
            if lidarState == RLParam.LEVEL_OF_RAY_CASTING.FAILED_DISTANCE:
                return -100
            elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.DANGEROUS_DISTANCE and currActionIndex != RLParam.ACTIONS_INDEX.STOP:
                finalReward -= 6
            elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.DANGEROUS_DISTANCE and currActionIndex == RLParam.ACTIONS_INDEX.STOP:
                finalReward += 1
        
        # Car out of lane
        if centerState == RLParam.LEVEL_OF_LANE.MIDDLE:
            finalReward += 2
        elif centerState == RLParam.LEVEL_OF_LANE.RIGHT or centerState == RLParam.LEVEL_OF_LANE.LEFT:
            finalReward -= 1
        elif centerState == RLParam.LEVEL_OF_LANE.MOST_RIGHT or centerState == RLParam.LEVEL_OF_LANE.MOST_LEFT:
            finalReward -= 10
            
        # Prevent stop and go back action
        if currActionIndex == RLParam.ACTIONS_INDEX.STOP:
            finalReward -= 5
        elif currActionIndex == RLParam.ACTIONS_INDEX.DESC_FORWARD_VELO:
            finalReward -= 5

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
            # TODO: write get initState
            state = "START STATE"
            totalReward = 0
            alpha = alphas[e]
            startTime = time.time()

            for _ in range(RLParam.MAX_EPISODE_STEPS):
                actionIndex = self._epsilonGreedyPolicy(currState=state)
                nextState, reward, done = act(state, actionIndex)
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


RL = RLAlgorithm(range(90), [1, 2, 3, 4])
print(RL.signalPerAreaData)
state = RL._hashFromDistanceToState()
print(state)
RL.getReward(state, 0)