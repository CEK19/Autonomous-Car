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
TOPIC_NAME_TRAFFIC_SIGN = rospy.get_param('TOPIC_NAME_TRAFFIC_SIGN')
TOPIC_NAME_AVOIDANCE = rospy.get_param('TOPIC_NAME_AVOIDANCE')
TOPIC_NAME_LANE_DETECTION = rospy.get_param('TOPIC_NAME_LANE_DETECTION')

# NODE_NAME_AVOIDANCE = "avoidance_node_name"
# TOPIC_NAME_VELOCITY = "/cmd_vel"
# TOPIC_NAME_LIDAR = "/scan"
# TOPIC_NAME_AVOIDANCE = "avoidance_topic"
# TOPIC_NAME_LANE_DETECTION = "lane_detection_topic"

LIDAR_MAX_RANGE = 3.5  # metters, unit
WIDTH_SIMULATE_MAP = int(2*LIDAR_MAX_RANGE*50)
HEIGH_SIMULATE_MAP = int(2*LIDAR_MAX_RANGE*50)
WIDTH_OPTIMAL_PATH = 50
HEIGH_OPTIMAL_PATH = 50
BLOCKED_COLOR = 255
NON_BLOCKED_COLOR = 0
LANE_THICKNESS = 2
DELTA = 12  # 50 / 14 / 10
DELTA_X = DELTA
DELTA_Y = DELTA
# for check space of goal is free
AREA_WIDTH = 4
AREA_HEIGHT = 10
NUM_POINTS_OF_DIRECTION = 8  # 35 / 12
MAX_STRAIGHT_VELOCITY = 0.05  # 0.05
MIN_STRAIGHT_VELOCITY = 0
MAX_TURN_VELOCITY = 0.5  # 2.0 # 1.0
MIN_TURN_VELOCITY = 0
MAGIC_NUMBER = 6

IMAGE_SAVED_PER_FRAME = 1
frameIndex = 0

LEFT_GOAL = 'left-goal'
RIGHT_GOAL = 'right-goal'
ANOTHER_GOAL = 'another-goal'

LOG_PATH = "/home/ubuntu/NCKH_workspace/KOT3_ws/src/kot3_pkg/scripts/imgs/log.txt"
IMG_PATH = "/home/ubuntu/NCKH_workspace/KOT3_ws/src/kot3_pkg/scripts/imgs/lidar/"

HAVE_DECISION_MAKING = False

pub = 0
if HAVE_DECISION_MAKING:
    pub = rospy.Publisher(TOPIC_NAME_AVOIDANCE, String, queue_size=1)
else:
    pub = rospy.Publisher(TOPIC_NAME_VELOCITY, Twist, queue_size=1)


class Utils:
    @staticmethod
    def writeLog(txt="", txt1="", txt2="", txt3="", txt4=""):
        log = open(LOG_PATH, "a")
        if len(txt) > 0:
            log.write(txt + " | ")
        if len(txt1) > 0:
            log.write(txt1 + " | ")
        if len(txt2) > 0:
            log.write(txt2 + " | ")
        if len(txt3) > 0:
            log.write(txt3 + " | ")
        if len(txt4) > 0:
            log.write(txt4 + " | ")
        log.write("\n")

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
    
    @staticmethod
    def rotate(image, angle, center = None, scale = 1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated


class CombineLidarLane:
    def __init__(self):
        self.lane_signal_subscriber = rospy.Subscriber(TOPIC_NAME_LANE_DETECTION, String, self.updateLaneDetectionSignal)
        self.lidar_signal_subscriber = rospy.Subscriber(TOPIC_NAME_LIDAR, LaserScan, self.updateLidarSignal)
        self.sign_signal_cubcriber = rospy.Subscriber(TOPIC_NAME_TRAFFIC_SIGN, String, self.updateSignSignal)
        self.ranges = [1 for _ in range(360)]
        self.tracePath = []
        self.sign = ""
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


    def drawLanesOnMap(self, simMap):
        leftBottomLaneXtmp = 0
        leftBottomLaneYtmp = 0
        leftTopLaneXtmp = 0
        leftTopLaneYtmp = 0
        rightBottomLaneXtmp = 0
        rightBottomLaneYtmp = 0
        rightTopLaneXtmp = 0
        rightTopLaneYtmp = 0

        
        if self.leftBottomLaneX == self.leftTopLaneX:
            leftBottomLaneXtmp = self.leftBottomLaneX
            leftBottomLaneYtmp = HEIGH_OPTIMAL_PATH-1
            leftTopLaneXtmp = self.leftTopLaneX
            leftTopLaneYtmp = 0
            simMap = cv2.line(simMap, (leftBottomLaneXtmp, leftBottomLaneYtmp), (leftTopLaneXtmp, leftTopLaneYtmp), BLOCKED_COLOR, LANE_THICKNESS)
        else:
            # Lane left: y = Ax + B
            A, B = Utils.getEquationOfLane([self.leftBottomLaneX, self.leftBottomLaneY], [self.leftTopLaneX, self.leftTopLaneY])
            
            leftBottomLaneXtmp = round((HEIGH_OPTIMAL_PATH-1-B)/A)
            leftBottomLaneYtmp = HEIGH_OPTIMAL_PATH-1
            leftTopLaneXtmp = round(-B/A)
            leftTopLaneYtmp = 0
            
            _, lbPoint, ltPoint = cv2.clipLine((0, 0, WIDTH_OPTIMAL_PATH-1, HEIGH_OPTIMAL_PATH-1), (leftBottomLaneXtmp, leftBottomLaneYtmp), (leftTopLaneXtmp, leftTopLaneYtmp))
            
            leftBottomLaneXtmp = lbPoint[0]
            leftBottomLaneYtmp = lbPoint[1]
            leftTopLaneXtmp = ltPoint[0]
            leftTopLaneYtmp = ltPoint[1]
            
            simMap = cv2.line(simMap, (leftBottomLaneXtmp, leftBottomLaneYtmp), (leftTopLaneXtmp, leftTopLaneYtmp), BLOCKED_COLOR, LANE_THICKNESS)
        
        if self.rightBottomLaneX == self.rightTopLaneX:
            rightBottomLaneXtmp = self.rightBottomLaneX
            rightBottomLaneYtmp = HEIGH_OPTIMAL_PATH-1
            rightTopLaneXtmp = self.rightTopLaneX
            rightTopLaneYtmp = 0
            simMap = cv2.line(simMap, (rightBottomLaneXtmp, rightBottomLaneYtmp), (rightTopLaneXtmp, rightTopLaneYtmp), BLOCKED_COLOR, LANE_THICKNESS)
        else:
            # Lane right: y = Cx + D
            C, D = Utils.getEquationOfLane([self.rightBottomLaneX, self.rightBottomLaneY], [self.rightTopLaneX, self.rightTopLaneY])
            
            rightBottomLaneXtmp = round((HEIGH_OPTIMAL_PATH-1-D)/C)
            rightBottomLaneYtmp = HEIGH_OPTIMAL_PATH-1
            rightTopLaneXtmp = round(-D/C)
            rightTopLaneYtmp = 0
            
            _, rbPoint, rtPoint = cv2.clipLine((0, 0, WIDTH_OPTIMAL_PATH-1, HEIGH_OPTIMAL_PATH-1), (rightBottomLaneXtmp, rightBottomLaneYtmp), (rightTopLaneXtmp, rightTopLaneYtmp))
            
            rightBottomLaneXtmp = rbPoint[0]
            rightBottomLaneYtmp = rbPoint[1]
            rightTopLaneXtmp = rtPoint[0]
            rightTopLaneYtmp = rtPoint[1]
            
            simMap = cv2.line(simMap, (rightBottomLaneXtmp, rightBottomLaneYtmp), (rightTopLaneXtmp, rightTopLaneYtmp), BLOCKED_COLOR, LANE_THICKNESS)


        self.goalX = (3*leftTopLaneXtmp+rightTopLaneXtmp)//4
        self.goalY = (3*leftTopLaneYtmp+rightTopLaneYtmp)//4
        self.goal2X = (leftTopLaneXtmp+3*rightTopLaneXtmp)//4
        self.goal2Y = (leftTopLaneYtmp+3*rightTopLaneYtmp)//4

        return simMap
    
    def updateSignSignal(self, msg):
        parsed = json.loads(msg.data)
        self.sign = parsed["sign"]

    def updateLaneDetectionSignal(self, msg):
        parsed = json.loads(msg.data)
        
        if parsed['bl'][1] == parsed['tl'][1]:
            return
        if parsed['br'][1] == parsed['tr'][1]:
            return

        self.leftBottomLaneX = parsed["bl"][0]
        self.leftBottomLaneY = parsed["bl"][1]
        self.leftTopLaneX = parsed["tl"][0]
        self.leftTopLaneY = parsed["tl"][1]
        self.rightTopLaneX = parsed["tr"][0]
        self.rightTopLaneY = parsed["tr"][1]
        self.rightBottomLaneX = parsed["br"][0]
        self.rightBottomLaneY = parsed["br"][1]
        self.frameIndex = parsed["frameIndex"]


    def updateLidarSignal(self, scan):
        # Front of robot index = 0, anti clockwise +1 index (left = 90 deg)
        self.ranges = scan.ranges

    def updateVelocity(self, event):
        if self.cmdFirstPub:
            self.cmdFirstPub = False
            self.cmdStartTime = time.time()
        self.cmdCount += 1
        myTwist = Twist()
        myTwist.linear.x = self.straightVel
        myTwist.angular.z = self.turnVel
        
        
        if HAVE_DECISION_MAKING:
            msg = json.dumps({"linear": self.straightVel, "angular": self.turnVel})
            pub.publish(msg)
        else:
            pub.publish(myTwist)

        # Utils.publicVelocity(self.straightVel, self.turnVel)

    def clearLineWhereGoalStuck(self, simulateMap):
        simulateMap[self.goalY:self.goalY + 2, :] = NON_BLOCKED_COLOR
        simulateMap[self.goal2Y:self.goal2Y + 2, :] = NON_BLOCKED_COLOR

    def chooseGoal(self, simulateMap):
        # check if only 1 goal feature
        if self.goal2X is None or self.goal2Y is None:
            Utils.writeLog("ChooseGoal func", "never appear", goalChoosen)
            return LEFT_GOAL, self.goalX, self.goalY

        # check free space above goals
        goalChoosen, curGoalX, curGoalY = self.chooseGoalByFreeSpaceAGoal(simulateMap)
        if goalChoosen is not ANOTHER_GOAL:
            Utils.writeLog("ChooseGoal func", "free space", goalChoosen)
            return goalChoosen, curGoalX, curGoalY

        # Go to another goal if 1 goal is blocked
        isGoal1Available = simulateMap[self.goalY, self.goalX] == NON_BLOCKED_COLOR
        isGoal2Available = simulateMap[self.goal2Y, self.goal2X] == NON_BLOCKED_COLOR
        if isGoal1Available and not isGoal2Available:
            Utils.writeLog("ChooseGoal func", "1 goal block", LEFT_GOAL)
            return LEFT_GOAL, self.goalX, self.goalY
        elif isGoal2Available and not isGoal1Available:
            Utils.writeLog("ChooseGoal func", "1 goal block", goalChoosen)
            return RIGHT_GOAL, self.goal2X, self.goal2Y
        elif isGoal1Available and isGoal2Available:
            # compare distance robot to each lanes
            goalChoosen, curGoalX, curGoalY = self.chooseGoalByDistanceRobotToLane()
            if goalChoosen is not ANOTHER_GOAL:
                Utils.writeLog("ChooseGoal func", "choose by distance both free", goalChoosen)
                return goalChoosen, curGoalX, curGoalY
        else:
            # both stuck
            self.clearLineWhereGoalStuck(simulateMap)


        # check free space above goals
        # goalChoosen, curGoalX, curGoalY = self.chooseGoalByFreeSpaceAGoal(simulateMap)
        # if goalChoosen is not ANOTHER_GOAL:
        #     return curGoalX, curGoalY

        # compare distance robot to each lanes
        goalChoosen, curGoalX, curGoalY = self.chooseGoalByDistanceRobotToLane()
        if goalChoosen is not ANOTHER_GOAL:
            Utils.writeLog("ChooseGoal func", "choose by distance both stuck", goalChoosen)
            return goalChoosen, curGoalX, curGoalY

        # Go on right lane at default
        Utils.writeLog("ChooseGoal func", "default", goalChoosen)
        return RIGHT_GOAL, self.goal2X, self.goal2Y

    def chooseGoalByDistanceRobotToGoal(self):
        curPoint = [WIDTH_OPTIMAL_PATH//2, HEIGH_OPTIMAL_PATH//2]

        d1 = Utils.distanceBetweenPoints(curPoint, [self.goalX, self.goalY])
        d2 = Utils.distanceBetweenPoints(curPoint, [self.goal2X, self.goal2Y])
        if d1 < d2:
            return LEFT_GOAL, self.goalX, self.goalY
        elif d1 > d2:
            return RIGHT_GOAL, self.goal2X, self.goal2Y
        return ANOTHER_GOAL, None, None
    
    def chooseGoalByDistanceRobotToLane(self):
        curPoint = [WIDTH_OPTIMAL_PATH//2, HEIGH_OPTIMAL_PATH//2]
        d1 = 0
        d2 = 0
        
        if (self.leftBottomLaneX == self.leftTopLaneX):
            d1 = abs(curPoint[0] - self.leftBottomLaneX)
        else:
            # left Lane: y = Ax + B
            A, B = Utils.getEquationOfLane([self.leftBottomLaneX, self.leftBottomLaneY], [self.leftTopLaneX, self.leftTopLaneY])
            d1 = Utils.getDistanceFromPointToLane(A, B, curPoint)
        
        if (self.rightBottomLaneX == self.rightTopLaneX):
            d2 = abs(curPoint[0] - self.rightBottomLaneX)
        else:
            # right Lane: y = Cx + D
            C, D = Utils.getEquationOfLane([self.rightBottomLaneX, self.rightBottomLaneY], [
                                            self.rightTopLaneX, self.rightTopLaneY])
            d2 = Utils.getDistanceFromPointToLane(C, D, curPoint)
        
        if d1 < d2:
            return LEFT_GOAL, self.goalX, self.goalY
        elif d1 > d2:
            return RIGHT_GOAL, self.goal2X, self.goal2Y
        return ANOTHER_GOAL, None, None


    def chooseGoalByFreeSpaceAGoal(self, simulateMap):
        leftGoalArea = simulateMap[self.goalX : self.goalX + AREA_WIDTH, self.goalY : self.goalY + AREA_HEIGHT]
        rightGoalArea = simulateMap[self.goal2X : self.goal2X - AREA_WIDTH, self.goal2Y : self.goal2Y + AREA_HEIGHT]

        isLeftGoalAreaBlocked = np.sum(leftGoalArea) / (BLOCKED_COLOR * AREA_WIDTH * AREA_HEIGHT) > 0.6
        isRightGoalAreaBlocked = np.sum(rightGoalArea) / (BLOCKED_COLOR * AREA_WIDTH * AREA_HEIGHT) > 0.6
        
        if isLeftGoalAreaBlocked and not isRightGoalAreaBlocked:
            return RIGHT_GOAL, self.goal2X, self.goal2Y
        elif not isLeftGoalAreaBlocked and isRightGoalAreaBlocked:
            return LEFT_GOAL, self.goalX, self.goalY
        return ANOTHER_GOAL, None, None
    
    def chooseGoalByFreeSpaceAGoalV2(self, simulateMap):
        leftGoalArea, rightGoalArea = 0, 0
        
        if self.leftBottomLaneX == self.leftTopLaneX:
            leftGoalArea = simulateMap[self.goalX : self.goalX + AREA_WIDTH, self.goalY : self.goalY + AREA_HEIGHT]
            rightGoalArea = simulateMap[self.goal2X : self.goal2X - AREA_WIDTH, self.goal2Y : self.goal2Y + AREA_HEIGHT]
        else:
            leftGoalArea = simulateMap[self.goalX : self.goalX + AREA_WIDTH, self.goalY : self.goalY + AREA_HEIGHT]
            rightGoalArea = simulateMap[self.goal2X : self.goal2X - AREA_WIDTH, self.goal2Y : self.goal2Y + AREA_HEIGHT]

        isLeftGoalAreaBlocked = np.sum(leftGoalArea) / (BLOCKED_COLOR * AREA_WIDTH * AREA_HEIGHT) > 0.6
        isRightGoalAreaBlocked = np.sum(rightGoalArea) / (BLOCKED_COLOR * AREA_WIDTH * AREA_HEIGHT) > 0.6
        
        if isLeftGoalAreaBlocked and not isRightGoalAreaBlocked:
            return RIGHT_GOAL, self.goal2X, self.goal2Y
        elif not isLeftGoalAreaBlocked and isRightGoalAreaBlocked:
            return LEFT_GOAL, self.goalX, self.goalY
        return ANOTHER_GOAL, None, None

    def rotatePoint(self):
        time_t = time.time()
        deltaTime = time_t - self.lastTimeReciveLane
        self.lastTimeReciveLane = time_t

        if self.turnVel == 0:
            deltaY = Utils.convertMetToPixel(self.straightVel) * deltaTime
            self.leftBottomLaneY = self.leftBottomLaneY + deltaY
            self.rightBottomLaneY = self.rightBottomLaneY + deltaY
            self.leftTopLaneY = self.leftTopLaneY + deltaY
            self.rightTopLaneY = self.rightTopLaneY + deltaY
            return

        R = abs(Utils.convertMetToPixel(self.straightVel) / self.turnVel)
        isRight = self.turnVel < 0
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


    def bestFirst(self, simulateMap, curGoalX, curGoalY):
        invertMap = cv2.bitwise_not(simulateMap)
        grid = Grid(matrix=invertMap)
        start = grid.node(WIDTH_OPTIMAL_PATH//2, HEIGH_OPTIMAL_PATH // 2)
        end = grid.node(curGoalX, curGoalY)

        path, _= self.pathFinder.find_path(start, end, grid)
        return path
        

    def pathFinding(self, event):
        try:
            global frameIndex
            Utils.writeLog("Frame index: " + str(frameIndex))
            ThinhTime = time.time()

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
            scaledLidarSignalBaseAngle = convertedLidarSignalBaseAngle * scaleFactor*np.sin(angleList)

            # Finding the scale distance param to display on image
            simulateMap = np.zeros(shape=(HEIGH_SIMULATE_MAP, WIDTH_SIMULATE_MAP), dtype='uint8')
            pathOnlyMap = np.zeros(shape=(HEIGH_SIMULATE_MAP, WIDTH_SIMULATE_MAP), dtype='uint8')
            coordinateYObstacleSimulationMap = HEIGH_SIMULATE_MAP//2 - scaledLidarSignalBaseAngle.astype(np.int16)
            coordinateXObstacleSimulationMap = WIDTH_SIMULATE_MAP//2 + ((scaledLidarSignalBaseAngle/(np.sin(angleList) + 0.0001)) * (np.cos(angleList))).astype(np.int16)
            filteredIndex = np.where(((coordinateYObstacleSimulationMap >= 0) & (coordinateYObstacleSimulationMap < HEIGH_SIMULATE_MAP) & (coordinateXObstacleSimulationMap >= 0) & (coordinateXObstacleSimulationMap < WIDTH_SIMULATE_MAP)))

            # Make obstacle bigger
            simulateMap[coordinateYObstacleSimulationMap[filteredIndex], coordinateXObstacleSimulationMap[filteredIndex]] = BLOCKED_COLOR

            # cv2.imshow("hinhSimulatQQ before", simulateMap)

            # Magic code to fix point at (width/2, height/2) is collision
            simulateMap[HEIGH_SIMULATE_MAP//2 - MAGIC_NUMBER: HEIGH_SIMULATE_MAP//2 + MAGIC_NUMBER, WIDTH_SIMULATE_MAP//2 - MAGIC_NUMBER: WIDTH_SIMULATE_MAP//2 + MAGIC_NUMBER] = NON_BLOCKED_COLOR

            tmpImg = Utils.imgInColor(simulateMap)
            tmpImg[HEIGH_SIMULATE_MAP//2 - MAGIC_NUMBER: HEIGH_SIMULATE_MAP//2 + MAGIC_NUMBER, WIDTH_SIMULATE_MAP//2 - MAGIC_NUMBER: WIDTH_SIMULATE_MAP//2 + MAGIC_NUMBER] = (255, 255, 0)
            # cv2.imshow("hinhSimulatQQ after", tmpImg)

            # Make obstacle bigger - Option 2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DELTA, DELTA))
            simulateMap = cv2.dilate(simulateMap, kernel)

            qqImg = Utils.imgInColor(simulateMap)
            qqImg[HEIGH_SIMULATE_MAP//2 - MAGIC_NUMBER: HEIGH_SIMULATE_MAP//2 + MAGIC_NUMBER, WIDTH_SIMULATE_MAP//2 - MAGIC_NUMBER: WIDTH_SIMULATE_MAP//2 + MAGIC_NUMBER] = (255, 255, 0)
            # cv2.imshow("big Map", simulateMap)

            # Cut the image in range 50px radius from robot (25px from left to 25px from right)
            simulateMap = simulateMap[WIDTH_SIMULATE_MAP//2 - WIDTH_OPTIMAL_PATH//2: WIDTH_SIMULATE_MAP//2 + WIDTH_OPTIMAL_PATH // 2, HEIGH_SIMULATE_MAP//2 - HEIGH_OPTIMAL_PATH//2: HEIGH_SIMULATE_MAP//2 + HEIGH_OPTIMAL_PATH//2]
            pathOnlyMap = pathOnlyMap[WIDTH_SIMULATE_MAP//2 - WIDTH_OPTIMAL_PATH//2: WIDTH_SIMULATE_MAP//2 + WIDTH_OPTIMAL_PATH // 2, HEIGH_SIMULATE_MAP//2 - HEIGH_OPTIMAL_PATH//2: HEIGH_SIMULATE_MAP//2 + HEIGH_OPTIMAL_PATH//2]

            # Draw path from lane detection in to map
            simulateMap = self.drawLanesOnMap(simulateMap)

            # Path finding here
            curGoalX = 0
            curGoalY = 0
            
            goalChoosen, curGoalX, curGoalY = self.chooseGoal(simulateMap)

            start_QTM = time.time()
            self.tracePath = self.bestFirst(simulateMap, curGoalX, curGoalY)
            
            if len(self.tracePath) == 0:
                if goalChoosen == LEFT_GOAL:
                    curGoalX = self.goal2X
                    curGoalY = self.goal2Y
                else:
                    curGoalX = self.goalX
                    curGoalY = self.goalY
                
                if simulateMap[curGoalY, curGoalX] == BLOCKED_COLOR:
                    self.clearLineWhereGoalStuck(simulateMap)
                Utils.writeLog("no path", "change to another goal", goalChoosen)
                self.tracePath = self.bestFirst(simulateMap, curGoalX, curGoalY)
            
            end_QTM = time.time()

            # draw trace path
            visualizedMap = cv2.cvtColor(simulateMap, cv2.COLOR_GRAY2RGB)
            if len(self.tracePath):
                for coor in self.tracePath:
                    x, y = coor[0], coor[1]
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

            print("**********************", time.time() - ThinhTime)
            
            # save image for bebug
            if frameIndex % IMAGE_SAVED_PER_FRAME == 0:
                cv2.imwrite(IMG_PATH + str(self.frameIndex) + "-" + str(frameIndex) + ".png", cv2.putText(visualizedMap, "F: " + str(self.frameIndex), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(255, 255, 0)))
            frameIndex += 1
            
            Utils.writeLog()

            if cv2.waitKey(1) == ord('q'):
                return
            
        except Exception as e:
            log = open(LOG_PATH, "a")
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

    def getPath(self):
        return self.tracePath

    def getVelocity(self):
        point15th = NUM_POINTS_OF_DIRECTION

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
        alpha = abs(angle - math.radians(90))

        straightVel = math.cos(alpha) * MAX_STRAIGHT_VELOCITY
        turnVel = math.sin(alpha) * MAX_TURN_VELOCITY

        if straightVel < 0 or turnVel < 0:
            print("fail at negative vel")

        # if straightVel < MIN_STRAIGHT_VELOCITY:
        #     straightVel = 0
        # if turnVel < MIN_TURN_VELOCITY:
        #     turnVel = 0
        
        if self.sign == "STOP":
            self.straightVel = 0
            self.turnVel = 0
            return

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
        # time.sleep(2)
        rospy.Timer(rospy.Duration(0.1), avoidance.pathFinding)  # 0.05
        rospy.Timer(rospy.Duration(0.1), avoidance.updateVelocity)  # 0.01

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
