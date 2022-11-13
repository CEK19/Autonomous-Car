from const import *
from utils import *
import random
import numpy as np
import json
import time
import math
import cv2


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
                for angle in RLParam.LEVEL_OF_ANGLE.LIST_LEVEL_ANGLES:
                    rs[combinedString + level + angle] = [0] * len(RLParam.ACTIONS)
                    for idx in range(len(RLParam.ACTIONS)):
                        rs[combinedString + level + angle][idx] = np.random.uniform(-10, 11)
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
            
            reward += (currYPos - preYPos)*RLParam.SCORE.INCREASE_Y
            reward += ((-currVelocity*math.cos(currAngle)) - (-preVelocity*math.cos(prevAngle)))*RLParam.SCORE.INCREASE_SPEED_FORWARD
            return reward

        def g(state):
            finalReward = 0
            stateArr = [char for char in state]
            lidarStates = stateArr[0:RLParam.AREA_RAY_CASTING_NUMBERS]
            centerState = stateArr[-1]
            
            currForwardVelocity = currentInfo['velocity']
            currAngle = currentInfo['angle']

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
            
            if (currForwardVelocity < 20):
                finalReward += RLParam.SCORE.STOPS_TO_ENJOY
            
            if (currAngle <= math.pi/2 or currAngle >= math.pi + math.pi/2):
                finalReward += abs(math.pi - currAngle)*RLParam.SCORE.TURN_AROUND
                
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
        
        #------------------------------------------ THINH BEDE ---------------------------
        visualMap = np.zeros((GameSettingParam.HEIGHT,GameSettingParam.WIDTH,3),dtype="uint8")
        outputVideo = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (GameSettingParam.WIDTH,GameSettingParam.HEIGHT))        
        #------------------------------------------ THINH BEDE ---------------------------
        
        for e in range(RLParam.N_EPISODES):
            state = env.getCurrentState()
            totalReward = 0
            alpha = alphas[e]
            epsilon = epsilons[e]
            startTime = time.time()
            
            #------------------------------------------ THINH BEDE ---------------------------
            # startPoint = (env.xPos, env.yPos)
            #------------------------------------------ THINH BEDE ---------------------------
            
            curEpochMap = np.zeros((GameSettingParam.HEIGHT,GameSettingParam.WIDTH,3),dtype="uint8")

            for actionCount in range(RLParam.MAX_EPISODE_STEPS):
                # print("state: ", state)
                actionIndex = self._epsilonGreedyPolicy(
                    currState=state, currentEpsilon=epsilon)
                nextState, reward, done = env.updateStateByAction(actionIndex)
                totalReward += reward
                self.Q[state][actionIndex] = self.Q[state][actionIndex] + \
                    alpha * (reward + RLParam.GAMMA *
                             np.max(self.Q[nextState]) - self.Q[state][actionIndex])
                state = nextState

#------------------------------------------ THINH BEDE ---------------------------#
                # curPoint = (int(env.xPos),int(env.yPos))
                # drawColor = (255,0,255)
                # if (int(actionCount/100)%2 == 0):
                #     drawColor = (0,255,0)
                # curEpochMap = cv2.line(curEpochMap,curPoint,startPoint,drawColor,2)
                # startPoint = curPoint
#------------------------------------------ THINH BEDE ---------------------------#       
        
                if done or actionCount == RLParam.MAX_EPISODE_STEPS - 1:
                    totalReward -= (actionCount + 1) * 0.01  # 120s * 1 = 120
                    break
            comment = f"Episode {e + 1}, xPos={env.xPos} - yPos={env.yPos} : total reward in {actionCount} actions -> {totalReward}\n"
            print(comment, end="")
            endTime = time.time()
            print(endTime - startTime)
            progressFile = open(FILE.PROGRESS, "a")
            progressFile.write(comment)
            progressFile.close()
            if e % 100  == 0 and not e == 0:
                print("--> start write to file")
                file = open(FILE.MODEL_SAVE, "w")
                file.write(json.dumps(self.Q))
                file.close()
                print("end write to file !!!")
                
            #------------------------------------------ THINH BEDE ---------------------------#              
            # cv2.addWeighted(visualMap,0.5,curEpochMap,1,0.0,visualMap)
            # visualMapWText = visualMap.copy()
            # visualMapWText = cv2.putText(visualMapWText,str(int(totalReward)),(20,GameSettingParam.HEIGHT - 50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(199,141,255),1,cv2.LINE_AA)
            # visualMapWText = cv2.putText(visualMapWText,str(e),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(199,141,255),1,cv2.LINE_AA)
            # visualMapWText = cv2.putText(visualMapWText,str(actionCount),(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(199,141,255),1,cv2.LINE_AA)
            # cv2.imshow("Last Path",visualMapWText)
            # outputVideo.write(visualMapWText)                
            # cv2.waitKey(1)
            #------------------------------------------ THINH BEDE ---------------------------#  
            
            env = env.reset()
