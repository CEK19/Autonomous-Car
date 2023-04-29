#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import importlib
import json
import numpy as np
import cv2
import heapq
import time
import math
import threading
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.best_first import BestFirst
from pathfinding.core.grid import Grid

# Test here

NODE_NAME_AVOIDANCE = rospy.get_param('NODE_NAME_AVOIDANCE')
TOPIC_NAME_VELOCITY = rospy.get_param('TOPIC_NAME_VELOCITY')
TOPIC_NAME_LIDAR = rospy.get_param('TOPIC_NAME_LIDAR')
TOPIC_NAME_AVOIDANCE = rospy.get_param('TOPIC_NAME_AVOIDANCE')
TOPIC_NAME_LANE_DETECTION = rospy.get_param('TOPIC_NAME_LANE_DETECTION')

# NODE_NAME_AVOIDANCE = "avoidance_node_name"
# TOPIC_NAME_VELOCITY = "/cmd_vel"
# TOPIC_NAME_LIDAR = "/scan"
# TOPIC_NAME_AVOIDANCE = "avoidance_topic"
# TOPIC_NAME_LANE_DETECTION = "lane_detection_topic"

LIDAR_MAX_RANGE = 3.5 # metters, unit
WIDTH_SIMULATE_MAP = int(2*LIDAR_MAX_RANGE*50)
HEIGH_SIMULATE_MAP = int(2*LIDAR_MAX_RANGE*50)
WIDTH_OPTIMAL_PATH = 50
HEIGH_OPTIMAL_PATH = 50
BLOCKED_COLOR = 255
NON_BLOCKED_COLOR = 0
LANE_THICKNESS = 2
DELTA = 10 # 50 / 14
DELTA_X = DELTA
DELTA_Y = DELTA
NUM_POINTS_OF_DIRECTION = 12 # 35 / 12
MAX_STRAIGHT_VELOCITY = 0.05  # 0.05
MAX_TURN_VELOCITY = 0.75  # 2.0 # 1.0

IMAGE_SAVED_PER_FRAME = 1
frameIndex = 0


# pub = rospy.Publisher(TOPIC_NAME_AVOIDANCE, String, queue_size=1)
pub = rospy.Publisher(TOPIC_NAME_VELOCITY, Twist, queue_size=1)

class Utils:
    @staticmethod
    def getVectorAB(A, B):
        return B[0] - A[0], B[1] - A[1]
    
    @staticmethod
    def getInvertVector(A):
        return -A[0], -A[1]
    
    @staticmethod
    def getRotatedPoint(point, center, angle):
        newX = center[0] + (point[0] - center[0]) * math.cos(angle) - (point[1] - center[1]) * math.sin(angle)
        newY = center[1] + (point[0] - center[0]) * math.sin(angle) + (point[1] - center[1]) * math.cos(angle)
        return newX, newY
    
    @staticmethod
    def distanceBetweenPoints(A, B):
        return math.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)
    
    @staticmethod
    def getAngleOfVectors(A, B):
        return np.arccos(np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B)))
    
    # y = Ax + b || Ax - y + b
    @staticmethod
    def isPointBetween2Lane(leftLaneA, leftLaneB, rightLaneA, rightLaneB, point):
        if point[1] > leftLaneA*point[0] + leftLaneB and point[1] > rightLaneA*point[0] + rightLaneB:
            return True
        return False
    
    # y = Ax + b || Ax - y + b
    @staticmethod
    def getDistanceFromPointToLane(A, B, point):
        return abs(A*point[0] - point[1] + B)/math.sqrt(A**2 + 1)
    
    # y = Ax + b || Ax - y + b
    @staticmethod
    def getEquationOfLane(point1, point2):
        A = (point1[1] - point2[1]) / (point1[0] - point2[0] + 0.0001)
        B = point1[1] - A*point1[0]
        return A, B
    
    # 3.5m = 350cm = 100pixels
    # 2*LIDAR_MAX_RANGE (m) = WIDTH_SIMULATE_MAP (pixel)
    @staticmethod
    def convertMetToPixel(m):
        return m*WIDTH_SIMULATE_MAP/(2*LIDAR_MAX_RANGE)
    
    @staticmethod
    def convertPixelToMet(pixel):
        return pixel*(2*LIDAR_MAX_RANGE)/WIDTH_SIMULATE_MAP
    
    @staticmethod
    def imgInColor(img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return imgRGB

    

class CombineLidarLane: 
    def __init__(self):
        self.lane_signal_subscriber = rospy.Subscriber(
            TOPIC_NAME_LANE_DETECTION, String, self.updateLaneDetectionSignal
        )
        self.lidar_signal_subscriber = rospy.Subscriber(
            TOPIC_NAME_LIDAR, LaserScan, self.updateLidarSignal)
        self.ranges = [1 for _ in range(360)]
        self.tracePath = []
        self.straightVel = 0
        self.turnVel = 0
        
        self.pathFinder = BestFirst(diagonal_movement=DiagonalMovement.always)

        # Goal of image
        self.goalX = WIDTH_OPTIMAL_PATH//2          # WIDTH_OPTIMAL_PATH//4
        self.goalY = 0
        self.goal2X = None                          # WIDTH_OPTIMAL_PATH*3//4
        self.goal2Y = None

        # Position of lane
        self.lastTimeReciveLane = time.time()
        self.leftBottomLaneX = 37
        self.leftBottomLaneY = HEIGH_OPTIMAL_PATH - 1

        self.leftTopLaneX = 37
        self.leftTopLaneY = 0

        self.rightTopLaneX = WIDTH_OPTIMAL_PATH - 1 - self.leftTopLaneX
        self.rightTopLaneY = 0

        self.rightBottomLaneX = WIDTH_OPTIMAL_PATH - 1 - self.leftBottomLaneX
        self.rightBottomLaneY = HEIGH_OPTIMAL_PATH - 1 
        
        # lane detection
        self.frameIndex = 0
        self.cmdFirstPub = True
        self.cmdStartTime = time.time()
        self.cmdCount = 0
    
    def printInfo(self):
        print("Bottom point: ", "[", self.leftBottomLaneX, self.leftBottomLaneY, "]", "[", self.rightBottomLaneX, self.rightBottomLaneY, "]")
        print("Top point: ", "[", self.leftTopLaneX, self.leftTopLaneY, "]", "[", self.rightTopLaneX, self.rightTopLaneY, "]")
        pass

    def drawLanesOnMap(self, simMap):
        # print("before", "[", self.leftBottomLaneX,self.leftBottomLaneY,"]", "[",self.leftTopLaneX,self.leftTopLaneY,"]", "[",self.rightBottomLaneX,self.rightBottomLaneY,"]","[",self.rightTopLaneX,self.rightTopLaneY,"]")
        # print("before",self.goalX,self.goalY,self.goal2X,self.goal2Y)
        color = 255
        
        # Lane left: y = Ax + B
        A, B = Utils.getEquationOfLane([self.leftBottomLaneX, self.leftBottomLaneY], [self.leftTopLaneX, self.leftTopLaneY])
        
        # Lane right: y = Cx + D
        C, D = Utils.getEquationOfLane([self.rightBottomLaneX, self.rightBottomLaneY], [self.rightTopLaneX, self.rightTopLaneY])
        
        leftBottomLaneXtmp = round((HEIGH_OPTIMAL_PATH-1-B)/(A+0.0001))
        leftBottomLaneYtmp  = HEIGH_OPTIMAL_PATH-1
        leftTopLaneXtmp  = round(-B/(A+0.0001))
        leftTopLaneYtmp  = 0
        rightTopLaneXtmp  = round(-D/(C+0.0001))
        rightTopLaneYtmp  = 0
        rightBottomLaneXtmp  = round((HEIGH_OPTIMAL_PATH-1-D)/(C+0.0001))
        rightBottomLaneYtmp  = HEIGH_OPTIMAL_PATH-1
        
        
        _, lbPoint, ltPoint = cv2.clipLine((0, 0, WIDTH_OPTIMAL_PATH-1, HEIGH_OPTIMAL_PATH-1), (leftBottomLaneXtmp , leftBottomLaneYtmp ), (leftTopLaneXtmp , leftTopLaneYtmp ))
        _, rbPoint, rtPoint = cv2.clipLine((0, 0, WIDTH_OPTIMAL_PATH-1, HEIGH_OPTIMAL_PATH-1), (rightBottomLaneXtmp , rightBottomLaneYtmp ), (rightTopLaneXtmp , rightTopLaneYtmp ))
        
        print("lbPoint", lbPoint)
        print("ltPoint", ltPoint)
        print("rbPoint", rbPoint)
        print("rtPoint", rtPoint)
        
        leftBottomLaneXtmp  = lbPoint[0]
        leftBottomLaneYtmp  = lbPoint[1]
        leftTopLaneXtmp  = ltPoint[0]
        leftTopLaneYtmp  = ltPoint[1]
        rightBottomLaneXtmp  = rbPoint[0]
        rightBottomLaneYtmp  = rbPoint[1]
        rightTopLaneXtmp  = rtPoint[0]
        rightTopLaneYtmp  = rtPoint[1]
        

        self.goalX = (3*leftTopLaneXtmp+rightTopLaneXtmp)//4
        self.goalY = (3*leftTopLaneYtmp+rightTopLaneYtmp)//4
        self.goal2X = (leftTopLaneXtmp+3*rightTopLaneXtmp)//4
        self.goal2Y = (leftTopLaneYtmp+3*rightTopLaneYtmp)//4

        simMap = cv2.line(simMap, (round(-B/(A+0.0001)), 0), (round((HEIGH_OPTIMAL_PATH-1-B)/(A+0.0001)), HEIGH_OPTIMAL_PATH-1), color, LANE_THICKNESS)
        simMap = cv2.line(simMap, (round(-D/(C+0.0001)), 0), (round((HEIGH_OPTIMAL_PATH-1-D)/(C+0.0001)), HEIGH_OPTIMAL_PATH-1), color, LANE_THICKNESS)
        return simMap

    def updateLaneDetectionSignal(self, msg):
        # print("msg", msg)
        parsed = json.loads(msg.data)
        # print("parsed", parsed)
        
        self.leftBottomLaneX = parsed["bl"][0]
        self.leftBottomLaneY = parsed["bl"][1]
        self.leftTopLaneX = parsed["tl"][0]
        self.leftTopLaneY = parsed["tl"][1]
        self.rightTopLaneX = parsed["tr"][0]
        self.rightTopLaneY = parsed["tr"][1]
        self.rightBottomLaneX = parsed["br"][0]
        self.rightBottomLaneY = parsed["br"][1]
        self.frameIndex = parsed["frameIndex"]
        print("frameIndex", parsed["frameIndex"])
        
        # self.goalX = WIDTH_SIMULATE_MAP//2
        # self.goalY = 0

    def updateLidarSignal(self, scan):
        # Front of robot index = 0, anti clockwise +1 index (left = 90 deg)
        self.ranges = scan.ranges

        # self.updateVelocity()
    
    def updateVelocity(self, event):
        if self.cmdFirstPub:
            self.cmdFirstPub = False
            self.cmdStartTime = time.time()
        self.cmdCount += 1
        print("cmd publish frequency: ", self.cmdCount/(time.time() - self.cmdStartTime))
        myTwist = Twist()
        myTwist.linear.x = self.straightVel
        myTwist.angular.z = self.turnVel
        pub.publish(myTwist)
        # msg = json.dumps({"linear": straight, "angular": angular})
        # pub.publish(msg)
        
        # Utils.publicVelocity(self.straightVel, self.turnVel)
        
    def rotatePoint(self):
        time_t = time.time()
        deltaTime = time_t - self.lastTimeReciveLane
        self.lastTimeReciveLane = time_t
        
        if self.turnVel == 0:
            print("plus toa do", deltaTime, self.straightVel,  Utils.convertMetToPixel(self.straightVel) * deltaTime)
            deltaY = Utils.convertMetToPixel(self.straightVel) * deltaTime
            self.leftBottomLaneY = self.leftBottomLaneY + deltaY
            self.rightBottomLaneY = self.rightBottomLaneY + deltaY
            self.leftTopLaneY = self.leftTopLaneY + deltaY
            self.rightTopLaneY = self.rightTopLaneY + deltaY
            return
        
        print("===== start rotate =====")
        # R = self.straightVel / self.turnVel
        R = abs(Utils.convertMetToPixel(self.straightVel) / self.turnVel)
        isRight = self.turnVel < 0
        # print("isRight: ", isRight, ", angular: ", self.turnVel, ", R: ", R, ", deltaTime: ", deltaTime)
        angular = self.turnVel * deltaTime
        alpha = abs(angular)
        curPos = [HEIGH_OPTIMAL_PATH // 2, WIDTH_OPTIMAL_PATH // 2]
        newPos = [0, 0]
        if isRight:
            newPos = [curPos[0] + 2*R*math.sin(alpha/2)*math.cos(math.pi/2 - alpha/2), curPos[1] - 2*R*math.sin(alpha/2)*math.sin(math.pi/2 - alpha/2)]
        else:
            newPos = [curPos[0] - 2*R*math.sin(alpha/2)*math.cos(math.pi/2 - alpha/2), curPos[1] - 2*R*math.sin(alpha/2)*math.sin(math.pi/2 - alpha/2)]
        vector = Utils.getVectorAB(curPos, newPos)
        invertVec = Utils.getInvertVector(vector)
        
        leftBottom = [self.leftBottomLaneX + invertVec[0], self.leftBottomLaneY + invertVec[1]]
        rightBottom = [self.rightBottomLaneX + invertVec[0], self.rightBottomLaneY + invertVec[1]]
        leftTop = [self.leftTopLaneX + invertVec[0], self.leftTopLaneY + invertVec[1]]
        rightTop = [self.rightTopLaneX + invertVec[0], self.rightTopLaneY + invertVec[1]]
        goal = [self.goalX + invertVec[0], self.goalY + invertVec[1]]
        goal2 = [self.goal2X + invertVec[0], self.goal2Y + invertVec[1]]
        
        
        self.leftBottomLaneX, self.leftBottomLaneY = Utils.getRotatedPoint(leftBottom, curPos, angular)
        self.rightBottomLaneX, self.rightBottomLaneY = Utils.getRotatedPoint(rightBottom, curPos, angular)
        self.leftTopLaneX, self.leftTopLaneY = Utils.getRotatedPoint(leftTop, curPos, angular)
        self.rightTopLaneX, self.rightTopLaneY = Utils.getRotatedPoint(rightTop, curPos, angular)
        self.goalX, self.goalY = Utils.getRotatedPoint(goal, curPos, angular)
        self.goal2X, self.goal2Y = Utils.getRotatedPoint(goal2, curPos, angular)

        cond1 = Utils.distanceBetweenPoints(curPos, leftBottom) == Utils.distanceBetweenPoints(curPos, [self.leftBottomLaneX, self.leftBottomLaneY])
        # if not cond1:
        #     print("fail rotate bottom left point", curPos, leftBottom, [self.leftBottomLaneX, self.leftBottomLaneY], math.degrees(angular))
        #     print(Utils.distanceBetweenPoints(curPos, leftBottom), Utils.distanceBetweenPoints(curPos, [self.leftBottomLaneX, self.leftBottomLaneY]))
            
        # cond2 = Utils.distanceBetweenPoints(curPos, rightBottom) == Utils.distanceBetweenPoints(curPos, [self.rightBottomLaneX, self.rightBottomLaneY])
        # if not cond2:
        #     print("fail rotate bottom right point", curPos, rightBottom, [self.rightBottomLaneX, self.rightBottomLaneY], angular)
        #     print(Utils.distanceBetweenPoints(curPos, rightBottom), Utils.distanceBetweenPoints(curPos, [self.rightBottomLaneX, self.rightBottomLaneY]))
            
        # cond3 = Utils.distanceBetweenPoints(curPos, leftTop) == Utils.distanceBetweenPoints(curPos, [self.leftTopLaneX, self.leftTopLaneY])
        # if not cond3:
        #     print("fail rotate top left point", curPos, leftTop, [self.leftTopLaneX, self.leftTopLaneY], angular)
        #     print(Utils.distanceBetweenPoints(curPos, leftTop), Utils.distanceBetweenPoints(curPos, [self.leftTopLaneX, self.leftTopLaneY]))
            
        # cond4 = Utils.distanceBetweenPoints(curPos, rightTop) == Utils.distanceBetweenPoints(curPos, [self.rightTopLaneX, self.rightTopLaneY])
        # if not cond4:
        #     print("fail rotate top right point", curPos, rightTop, [self.rightTopLaneX, self.rightTopLaneY], angular)
        #     print(Utils.distanceBetweenPoints(curPos, rightTop), Utils.distanceBetweenPoints(curPos, [self.rightTopLaneX, self.rightTopLaneY]))
            
        # cond5 = Utils.distanceBetweenPoints(curPos, goal) == Utils.distanceBetweenPoints(curPos, [self.goalX, self.goalY])
        # if not cond5:
        #     print("fail rotate goal point", curPos, goal, [self.goalX, self.goalY], angular)
        #     print(Utils.distanceBetweenPoints(curPos, goal), Utils.distanceBetweenPoints(curPos, [self.goalX, self.goalY]))
            
        # cond6 = Utils.distanceBetweenPoints(curPos, goal2) == Utils.distanceBetweenPoints(curPos, [self.goal2X, self.goal2Y])
        # if not cond6:
        #     print("fail rotate goal2 point", curPos, goal2, [self.goal2X, self.goal2Y], angular)
        #     print(Utils.distanceBetweenPoints(curPos, goal2), Utils.distanceBetweenPoints(curPos, [self.goal2X, self.goal2Y]))
        
        pass
                
    def pathFinding(self, event):
        try:
            # print("start at", time.time())
            self.rotatePoint()
            
            # Convert lidar distance signal, result represent left to right (0 -> 179)
            angleList = np.arange(start=0, stop=360, step=1, dtype=np.int16)
            angleList = angleList * np.pi/180.
            convertedLidarSignalBaseAngle = np.zeros(shape=((360, )))

            # 0 -> 180 in anti clockwise
            convertedLidarSignalBaseAngle[0:90] = self.ranges[270:360]
            convertedLidarSignalBaseAngle[90:180] = self.ranges[0:90]
            convertedLidarSignalBaseAngle[180:270] = self.ranges[90:180]
            convertedLidarSignalBaseAngle[270:360] = self.ranges[180:270]
            
            # Convert distance signal into vertical axis, then find the scale factor
            scaleFactor = HEIGH_SIMULATE_MAP/(2*LIDAR_MAX_RANGE)

            # Projected onto the Y axis
            scaledLidarSignalBaseAngle = convertedLidarSignalBaseAngle*scaleFactor*np.sin(angleList)

            # Finding the scale distance param to display on image
            simulateMap = np.zeros(shape=(HEIGH_SIMULATE_MAP, WIDTH_SIMULATE_MAP), dtype='uint8')
            pathOnlyMap = np.zeros(shape=(HEIGH_SIMULATE_MAP, WIDTH_SIMULATE_MAP), dtype='uint8')
            coordinateYObstacleSimulationMap = HEIGH_SIMULATE_MAP//2 - scaledLidarSignalBaseAngle.astype(np.int16) 
            coordinateXObstacleSimulationMap = WIDTH_SIMULATE_MAP//2 + ((scaledLidarSignalBaseAngle/(np.sin(angleList) + 0.0001))*(np.cos(angleList))).astype(np.int16)        
            filteredIndex = np.where(((coordinateYObstacleSimulationMap >= 0) & (coordinateYObstacleSimulationMap < HEIGH_SIMULATE_MAP) & (coordinateXObstacleSimulationMap >= 0) & (coordinateXObstacleSimulationMap < WIDTH_SIMULATE_MAP)))
            
            # Make obstacle bigger
            simulateMap[coordinateYObstacleSimulationMap[filteredIndex], coordinateXObstacleSimulationMap[filteredIndex]] = BLOCKED_COLOR
            
            cv2.imshow("hinhSimulatQQ before", simulateMap)
            
            # Magic code to fix point at (width/2, height/2) is collision
            simulateMap[HEIGH_SIMULATE_MAP//2 - 6 : HEIGH_SIMULATE_MAP//2 + 6, WIDTH_SIMULATE_MAP//2 - 6 : WIDTH_SIMULATE_MAP//2 + 6] = NON_BLOCKED_COLOR
            
            tmpImg = Utils.imgInColor(simulateMap)
            tmpImg[HEIGH_SIMULATE_MAP//2 - 6 : HEIGH_SIMULATE_MAP//2 + 6, WIDTH_SIMULATE_MAP//2 - 6 : WIDTH_SIMULATE_MAP//2 + 6] = (255, 255, 0)
            cv2.imshow("hinhSimulatQQ after", tmpImg)
            
            # Make obstacle bigger - Option 2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DELTA, DELTA))
            simulateMap = cv2.dilate(simulateMap, kernel)

            qqImg = Utils.imgInColor(simulateMap)
            qqImg[HEIGH_SIMULATE_MAP//2 - 6 : HEIGH_SIMULATE_MAP//2 + 6, WIDTH_SIMULATE_MAP//2 - 6 : WIDTH_SIMULATE_MAP//2 + 6] = (255, 255, 0)
            cv2.imshow("big Map", simulateMap)
            
            # Cut the image in range 50px radius from robot (25px from left to 25px from right)
            simulateMap = simulateMap[WIDTH_SIMULATE_MAP//2 - WIDTH_OPTIMAL_PATH//2 : WIDTH_SIMULATE_MAP//2 + WIDTH_OPTIMAL_PATH//2, HEIGH_SIMULATE_MAP//2 - HEIGH_OPTIMAL_PATH//2 : HEIGH_SIMULATE_MAP//2 + HEIGH_OPTIMAL_PATH//2]
            pathOnlyMap = pathOnlyMap[WIDTH_SIMULATE_MAP//2 - WIDTH_OPTIMAL_PATH//2 : WIDTH_SIMULATE_MAP//2 + WIDTH_OPTIMAL_PATH//2, HEIGH_SIMULATE_MAP//2 - HEIGH_OPTIMAL_PATH//2 : HEIGH_SIMULATE_MAP//2 + HEIGH_OPTIMAL_PATH//2]
            
            # Draw path from lane detection in to map
            simulateMap = self.drawLanesOnMap(simulateMap)

            # Path finding here
            # initalX, initialY = WIDTH_SIMULATE_MAP//2, HEIGH_SIMULATE_MAP - 1

            # If not exist lane to go -> do nothing
            rowAllBlocked = np.all(simulateMap == BLOCKED_COLOR, axis=1)
            # isExistLaneToGo = not np.any(rowAllBlocked == True)
            
            
            # PathFinding with library approach
            curGoalX = 0
            curGoalY = 0
            
            # Choose current goal from 2 goals
            isGoal1Available = simulateMap[self.goalY, self.goalX] == NON_BLOCKED_COLOR
            isGoal2Available = self.goal2X is not None and self.goal2Y is not None and simulateMap[self.goal2Y, self.goal2X] == NON_BLOCKED_COLOR
            
            if not isGoal1Available and not isGoal2Available:
                if self.goal2Y is None:
                    simulateMap[self.goalY, :] = NON_BLOCKED_COLOR
                elif self.goalY < self.goal2Y:
                    simulateMap[self.goalY, :] = NON_BLOCKED_COLOR
                else:
                    simulateMap[self.goal2Y, :] = NON_BLOCKED_COLOR
                    
            
            if isGoal1Available and not isGoal2Available:
                curGoalX, curGoalY = self.goalX, self.goalY
            elif isGoal2Available and not isGoal1Available:
                curGoalX, curGoalY = self.goal2X, self.goal2Y
            else:
                # find shortest path
                curPoint = [WIDTH_OPTIMAL_PATH//2, HEIGH_OPTIMAL_PATH//2]
                if self.goal2X is None or self.goal2Y is None:
                    curGoalX, curGoalY = self.goalX, self.goalY
                else:
                    # left Lane: y = Ax + B
                    A, B = Utils.getEquationOfLane([self.leftBottomLaneX, self.leftBottomLaneY], [self.leftTopLaneX, self.leftTopLaneY])
                    # right Lane: y = Cx + D
                    C, D = Utils.getEquationOfLane([self.rightBottomLaneX, self.rightBottomLaneY], [self.rightTopLaneX, self.rightTopLaneY])
                    
                    # d1 = Utils.distanceBetweenPoints([self.goalX, self.goalY], curPoint)
                    # d2 = Utils.distanceBetweenPoints([self.goal2X, self.goal2Y], curPoint)
                    d1 = Utils.getDistanceFromPointToLane(A, B, curPoint)
                    d2 = Utils.getDistanceFromPointToLane(C, D, curPoint)
                    if d1 <= d2:
                        curGoalX, curGoalY = self.goalX, self.goalY
                    else:
                        curGoalX, curGoalY = self.goal2X, self.goal2Y
            
            invertMap = cv2.bitwise_not(simulateMap)
            grid = Grid(matrix=invertMap)
            start = grid.node(WIDTH_OPTIMAL_PATH//2, HEIGH_OPTIMAL_PATH // 2)
            print(curGoalX, curGoalY)
            end = grid.node(curGoalX, curGoalY)
            
            start_QTM = time.time()
            self.tracePath, _ = self.pathFinder.find_path(start, end, grid)
            end_QTM = time.time()
            print(end_QTM - start_QTM)
            # print("Path: ", self.tracePath)
            
            # draw trace path
            visualizedMap = cv2.cvtColor(simulateMap, cv2.COLOR_GRAY2RGB)
            if len(self.tracePath):
                for coor in self.tracePath:
                    x, y = coor[0], coor[1]
                    # simulateMap =  cv2.circle(simulateMap, (x, y), 1, (255, 0, 0), 1)
                    pathOnlyMap[y, x] = 255
                    visualizedMap[y, x] = (0, 0, 255)
                    simulateMap[y, x] = 255

            # draw goal
            cv2.circle(visualizedMap, (self.goalX, self.goalY), 1, (0, 255, 0), 2)
            cv2.circle(visualizedMap, (self.goal2X, self.goal2Y), 1, (255, 255, 0), 2)

            visualizedMap = cv2.resize(visualizedMap, (WIDTH_OPTIMAL_PATH*4, HEIGH_OPTIMAL_PATH*4))
            cv2.imshow("simulate map", visualizedMap)
            # cv2.imshow("simulate map", cv2.resize(simulateMap, (WIDTH_OPTIMAL_PATH*4, HEIGH_OPTIMAL_PATH*4)))
            # cv2.imshow("path only map", cv2.resize(pathOnlyMap, (WIDTH_OPTIMAL_PATH*4, HEIGH_OPTIMAL_PATH*4)))
            
            self.getVelocity()
            # print("end at", time.time())
        
            global frameIndex
            if frameIndex % IMAGE_SAVED_PER_FRAME == 0:
                cv2.imwrite("/home/minhtu/NCKH_workspace/KOT3_ws/src/kot3_pkg/scripts/imgs/lidar/" + str(frameIndex) + "-" + str(self.frameIndex) + ".png", cv2.putText(visualizedMap, "F: " + str(self.frameIndex),(10, 25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(255, 255, 0)))
            frameIndex += 1
            
            if cv2.waitKey(1) == ord('q'):
                return   
        except Exception as e:
            log = open("/home/minhtu/NCKH_workspace/KOT3_ws/src/kot3_pkg/scripts/imgs/log.txt", "a")
            log.write(e)
            log.write("\n")
            print(e)
            pass

    def sendActionToTopic(self, action):
        """
        Input: action
        Ouput: None
        """
        message = json.dumps({"action": action})
        self.avoidance_publisher.publish(message)

    # def solve(self, data):
    #     self.sendActionToTopic(action)
    #     rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    def getPath(self):
        return self.tracePath
    
    def getVelocity(self):
        # point15th = len(self.tracePath) - 1 - NUM_POINTS_OF_DIRECTION
        point15th = NUM_POINTS_OF_DIRECTION
        # print(self.tracePath)
        if (point15th < 0 or not len(self.tracePath)):
            self.straightVel = 0
            self.turnVel = 0
            return
        if point15th >= len(self.tracePath):
            point15th = len(self.tracePath) - 1
        vecDirection = Utils.getVectorAB([HEIGH_OPTIMAL_PATH//2, WIDTH_OPTIMAL_PATH // 2], self.tracePath[point15th])
        vecZero = (1, 0)
        angle = Utils.getAngleOfVectors(vecDirection, vecZero)
        isRight = angle < math.radians(90)
        # isCenter = math.radians(87) < angle < math.radians(93)
        alpha = abs(angle - math.radians(90))
        straightVel = math.cos(alpha) * MAX_STRAIGHT_VELOCITY
        turnVel = math.sin(alpha) * MAX_TURN_VELOCITY
        
        
        if straightVel < 0 or turnVel < 0:
            print("fail at negative vel")
        
        if isRight:
            self.straightVel = straightVel    # straightVel
            self.turnVel = -turnVel
            return
        
        self.straightVel = straightVel    # straightVel
        self.turnVel = turnVel


if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_AVOIDANCE, anonymous=True)
        avoidance = CombineLidarLane()
        time.sleep(2)
        rospy.Timer(rospy.Duration(0.2), avoidance.pathFinding) # 0.05
        rospy.Timer(rospy.Duration(0.1), avoidance.updateVelocity) # 0.01
        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass