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
                        for yVelo in RLParam.LEVEL_OF_Y_VELO.LIST_LEVEL_OF_Y_VELO:
                            rs[combinedString + level + angle + omega + yVelo] = [0] * len(RLParam.ACTIONS)
                            for idx in range(len(RLParam.ACTIONS)):
                                rs[combinedString + level + angle + omega + yVelo][idx] = np.random.uniform(-10, 11)
        return rs

    # @staticmethod
    # def convertRayCastingDataToSignalPerArea(rayCastingData):
    #     # print("rayCastingData",rayCastingData)
    #     div = len(rayCastingData) // RLParam.AREA_RAY_CASTING_NUMBERS
    #     mod = len(rayCastingData) % RLParam.AREA_RAY_CASTING_NUMBERS

    #     tmpData = [0] * (RLParam.AREA_RAY_CASTING_NUMBERS +
    #                      (0 if mod == 0 else 1))
        
    #     tmpCount = 0
    #     for i in range(len(tmpData)):
    #         tmp = []
    #         for _ in range(div):
    #             tmp.append(rayCastingData[tmpCount])
    #             tmpCount += 1
    #             if (tmpCount == len(rayCastingData)):
    #                 break
    #         tmpData[i] = min(tmp)
    #     # print("tmpData",tmpData)
    #     return tmpData
    
    @staticmethod
    def convertRayCastingDataToSignalPerArea(rayCastingData):
        tmpData = [0] * 4
        
        tmpCount = 0
        for i in range(len(tmpData)):
            tmp = []
            for _ in range(RLParam.LEVEL_OF_RAY_CASTING.LIST_OF_ANGLE[i]):
                tmp.append(rayCastingData[tmpCount])
                tmpCount += 1
                if tmpCount == len(rayCastingData):
                    break
            tmpData[i] = min(tmp)
        return tmpData

    @staticmethod
    def hashFromDistanceToState(signalPerAreaData, leftSideDistance, rightSideDistance, angle, RotationVelocity, yVelo):
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
        else:
            hashFromAngle += RLParam.LEVEL_OF_ANGLE.OVER_ROTATION            
    
        hashFromOmega = ""
        if RotationVelocity >= RLParam.LEVEL_OF_ROTATION.MAX_RIGHT_ANGLE:
            hashFromOmega = RLParam.LEVEL_OF_ROTATION.MAX_RIGHT
        elif RotationVelocity >= RLParam.LEVEL_OF_ROTATION.RIGHT_ANGLE:
            hashFromOmega = RLParam.LEVEL_OF_ROTATION.RIGHT
        elif RotationVelocity <= RLParam.LEVEL_OF_ROTATION.MAX_LEFT_ANGLE:
            hashFromOmega = RLParam.LEVEL_OF_ROTATION.MAX_LEFT
        elif RotationVelocity <= RLParam.LEVEL_OF_ROTATION.LEFT_ANGLE:
            hashFromOmega = RLParam.LEVEL_OF_ROTATION.LEFT
        else:
            hashFromOmega = RLParam.LEVEL_OF_ROTATION.CENTER
            
        hashFromYVelo = ""
        if yVelo < RLParam.LEVEL_OF_Y_VELO.BACKWARD_VEL:
            hashFromYVelo = RLParam.LEVEL_OF_Y_VELO.BACKWARD
        elif yVelo < RLParam.LEVEL_OF_Y_VELO.STOP_VEL:
            hashFromYVelo = RLParam.LEVEL_OF_Y_VELO.STOP
        elif yVelo < RLParam.LEVEL_OF_Y_VELO.FORWARD_VEL:
            hashFromYVelo = RLParam.LEVEL_OF_Y_VELO.FORWARD
        elif yVelo <= RLParam.LEVEL_OF_Y_VELO.FAST_FORWARD_VEL:
            hashFromYVelo = RLParam.LEVEL_OF_Y_VELO.FAST_FORWARD
        
        return hashFromRayCasting + hashFromCenterOfLane + hashFromAngle + hashFromOmega + hashFromYVelo
    
    @staticmethod
    def decodeStateToName(stateNumber, type):
        if type == "lidar":
            if stateNumber == RLParam.LEVEL_OF_RAY_CASTING.INFINITY:
                return "infinity"
            elif stateNumber == RLParam.LEVEL_OF_RAY_CASTING.FAR_DISTANCE:
                return "far"
            elif stateNumber == RLParam.LEVEL_OF_RAY_CASTING.SAFETY_DISTANCE:
                return "safe"
            elif stateNumber == RLParam.LEVEL_OF_RAY_CASTING.DANGEROUS_DISTANCE:
                return "dangerous"
            elif stateNumber == RLParam.LEVEL_OF_RAY_CASTING.FAILED_DISTANCE:
                return "failed"
        elif type == "centerOfLane":
            if stateNumber == RLParam.LEVEL_OF_LANE.MIDDLE:
                return "middle"
            elif stateNumber == RLParam.LEVEL_OF_LANE.LEFT:
                return "left"
            elif stateNumber == RLParam.LEVEL_OF_LANE.RIGHT:
                return "right"
            elif stateNumber == RLParam.LEVEL_OF_LANE.MOST_LEFT:
                return "most-left"
            elif stateNumber == RLParam.LEVEL_OF_LANE.MOST_RIGHT:
                return "most-right"
        elif type == "angle":
            if stateNumber == RLParam.LEVEL_OF_ANGLE.FRONT:
                return "front"
            elif stateNumber == RLParam.LEVEL_OF_ANGLE.NORMAL_LEFT:
                return "normal-left"
            elif stateNumber == RLParam.LEVEL_OF_ANGLE.NORMAL_RIGHT:
                return "normal-right"
            elif stateNumber == RLParam.LEVEL_OF_ANGLE.OVER_ROTATION:
                return "over-rotation"
        elif type == "omega":
            if stateNumber == RLParam.LEVEL_OF_ROTATION.CENTER:
                return "center - no turning"
            elif stateNumber == RLParam.LEVEL_OF_ROTATION.LEFT:
                return "rotate-left"
            elif stateNumber == RLParam.LEVEL_OF_ROTATION.MAX_LEFT:
                return "rotate-max-left"
            elif stateNumber == RLParam.LEVEL_OF_ROTATION.RIGHT:
                return "rotate-right"
            elif stateNumber == RLParam.LEVEL_OF_ROTATION.MAX_RIGHT:
                return "rotate-max-right"
        elif type == "yVelo":
            if stateNumber == RLParam.LEVEL_OF_Y_VELO.FAST_FORWARD:
                return "fast-forward"
            elif stateNumber == RLParam.LEVEL_OF_Y_VELO.FORWARD:
                return "forward"
            elif stateNumber == RLParam.LEVEL_OF_Y_VELO.BACKWARD:
                return "backward"

    @staticmethod
    def getReward(currState, previousInfo, currentInfo, actionIndex):
        # reward = f_t - f_(t-1) + g(t)
        # def f(past, current):
        #     reward = 0
            
        #     preVelocity = past["velocity"]
        #     prevAngle = past["angle"]
        #     preYPos = past["yPos"]
            
        #     currVelocity = current["velocity"]            
        #     currAngle = current["angle"]
        #     currYPos = current["yPos"]
            
        #     if currYPos < 10:
        #         reward += RLParam.SCORE.FINISH_LINE
        #     elif currYPos < GameSettingParam.HEIGHT * (1 / 4):
        #         reward += 10000
        #     elif currYPos < GameSettingParam.HEIGHT * (2 / 4):
        #         reward += 1000
        #     elif currYPos < GameSettingParam.HEIGHT * (3 / 4):
        #         reward += 100
        #     else: 
        #         reward += 10
        #     # reward += (currYPos - preYPos)*RLParam.SCORE.INCREASE_Y
        #     # reward += ((-currVelocity*math.cos(currAngle)) - (-preVelocity*math.cos(prevAngle)))*RLParam.SCORE.INCREASE_SPEED_FORWARD
        #     return reward

        def g(state, actionIndex):
            finalReward = 0
            stateArr = [char for char in state]
            print(state)
            lidarStates = stateArr[0:RLParam.AREA_RAY_CASTING_NUMBERS]
            yVeloState = stateArr[-1]
            omagaState = stateArr[-2]
            angleState = stateArr[-3]
            centerState = stateArr[-4]
            
            print(f"===> gain state: lidar [{RLAlgorithm.decodeStateToName(lidarStates[0], 'lidar')}, {RLAlgorithm.decodeStateToName(lidarStates[1], 'lidar')}, {RLAlgorithm.decodeStateToName(lidarStates[2], 'lidar')}, {RLAlgorithm.decodeStateToName(lidarStates[3], 'lidar')}]")
            print(f"                 centerState: {RLAlgorithm.decodeStateToName(centerState, 'centerOfLane')}")
            print(f"                 angleState: {RLAlgorithm.decodeStateToName(angleState, 'angle')}")
            print(f"                 omegaState: {RLAlgorithm.decodeStateToName(omagaState, 'omega')}")
            print(f"                 yVelo: {RLAlgorithm.decodeStateToName(yVeloState, 'yVelo')}")

            
            # currForwardVelocity = currentInfo['velocity']
            # currAngle = currentInfo['angle']

            # Obstacles block car
            for index, lidarState in enumerate(lidarStates):
                tempReward = 0
                # Center ray
                if index == 1 or index == 2:
                    if lidarState == RLParam.LEVEL_OF_RAY_CASTING.INFINITY:
                        tempReward = 0
                    elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.FAR_DISTANCE:
                        tempReward = -2
                    elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.SAFETY_DISTANCE:
                        tempReward = -25
                    elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.DANGEROUS_DISTANCE:
                        tempReward = -80
                    elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.FAILED_DISTANCE:
                        tempReward = -1000
                
                # other ray
                else:
                    if lidarState == RLParam.LEVEL_OF_RAY_CASTING.DANGEROUS_DISTANCE:
                        tempReward = -40
                    elif lidarState == RLParam.LEVEL_OF_RAY_CASTING.FAILED_DISTANCE:
                        tempReward = -1000
                finalReward += tempReward
            
            # Angle of car
            if angleState == RLParam.LEVEL_OF_ANGLE.FRONT:
                finalReward += 3
            elif angleState == RLParam.LEVEL_OF_ANGLE.NORMAL_LEFT or angleState == RLParam.LEVEL_OF_ANGLE.NORMAL_RIGHT:
                finalReward += -5
            elif angleState == RLParam.LEVEL_OF_ANGLE.OVER_ROTATION:
                # if lidarStates[0] != RLParam.LEVEL_OF_RAY_CASTING.INFINITY and lidarStates[0] != RLParam.LEVEL_OF_RAY_CASTING.FAR_DISTANCE and \
                #    (lidarStates[1] != RLParam.LEVEL_OF_RAY_CASTING.INFINITY or \
                #    lidarStates[2] != RLParam.LEVEL_OF_RAY_CASTING.INFINITY) and \
                #    lidarStates[3] != RLParam.LEVEL_OF_RAY_CASTING.INFINITY and lidarStates[3] != RLParam.LEVEL_OF_RAY_CASTING.FAR_DISTANCE:
                #     finalReward += 0
                finalReward += -700
                    
            # Car out of lane
            if centerState == RLParam.LEVEL_OF_LANE.MIDDLE:
                finalReward += 2
            elif centerState == RLParam.LEVEL_OF_LANE.RIGHT or centerState == RLParam.LEVEL_OF_LANE.LEFT:
                finalReward += -2
            elif centerState == RLParam.LEVEL_OF_LANE.MOST_RIGHT or centerState == RLParam.LEVEL_OF_LANE.MOST_LEFT:
                finalReward += -100
                
            # yVelo
            if yVeloState == RLParam.LEVEL_OF_Y_VELO.FAST_FORWARD:
                infState = RLParam.LEVEL_OF_RAY_CASTING.INFINITY
                farState = RLParam.LEVEL_OF_RAY_CASTING.FAR_DISTANCE
                if (lidarStates[0] == infState or lidarStates[0] == farState) and lidarStates[1] == infState and lidarStates[2] == infState and (lidarStates[3] == infState or lidarStates[3] == farState):
                    finalReward += 10
                finalReward += 3
            elif yVeloState == RLParam.LEVEL_OF_Y_VELO.FORWARD:
                finalReward += 0

            
            # action
            if RLParam.ACTIONS[actionIndex] == PlayerParam.STOP:
                finalReward += -5
            elif RLParam.ACTIONS[actionIndex] == PlayerParam.ACCELERATION_FORWARD:
                finalReward += 2
            
                
            return finalReward

        # totalReward = f(past=previousInfo, current=currentInfo) + g(state=currState)
        totalReward = g(state=currState, actionIndex=actionIndex)
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
            keys = pygame.key.get_pressed()
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
                print("state:", state)

                if done or actionCount == RLParam.MAX_EPISODE_STEPS - 1: 
                    # totalReward -=  (actionCount + 1) * 0.01 # 120s * 1 = 120
                    break
                # print("step time: ",time.time() - startTime)
            comment = f"Episode {e + 1}, xPos={env.xPos} - yPos={env.yPos} : total reward in {actionCount} actions -> {totalReward}\n"
            
            print(comment)

            if (e%RLParam.SAVE_PER_EPISODE == 0 and e != 0) or (e == RLParam.N_EPISODES - 1):
                print("save file")
                file = open("rl-learning.txt", "w")
                file.write(json.dumps(self.Q))
                file.close()
            
            progressFile = open("progress.txt", "a")
            progressFile.write(comment)
            
            if (keys[pygame.K_p]):
                print("save file and exit")
                file = open("rl-learning.txt", "w")
                file.write(json.dumps(self.Q))
                file.close()
                break
            
            env = env.reset()
