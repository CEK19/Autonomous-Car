#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import importlib
import json
import numpy as np
import cv2
import heapq
import time
import math
import threading
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Test here

NODE_NAME_AVOIDANCE = rospy.get_param('NODE_NAME_AVOIDANCE')
TOPIC_NAME_VELOCITY = rospy.get_param('TOPIC_NAME_VELOCITY')
TOPIC_NAME_LIDAR = rospy.get_param('TOPIC_NAME_LIDAR')
TOPIC_NAME_AVOIDANCE = rospy.get_param('TOPIC_NAME_AVOIDANCE')
TOPIC_NAME_LANE_DETECTION = rospy.get_param('TOPIC_NAME_LANE_DETECTION')

LIDAR_MAX_RANGE = 3  # metters, unit
WIDTH_SIMULATE_MAP = 2*LIDAR_MAX_RANGE*100
HEIGH_SIMULATE_MAP = LIDAR_MAX_RANGE*100
BLOCKED_COLOR = 255
DELTA = 50
DELTA_X = DELTA
DELTA_Y = DELTA
NUM_POINTS_OF_DIRECTION = 35  # 27
MAX_STRAIGHT_VELOCITY = 0.2  # 0.2
MAX_TURN_VELOCITY = 2.0  # 1.4


pub = rospy.Publisher(TOPIC_NAME_AVOIDANCE, String, queue_size=1)


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
        newY = center[1] + (point[0] - center[0]) * math.sin(angle) + (point[1] + center[1]) * math.cos(angle)
        return

    @staticmethod
    def getAngleOfVectors(A, B):
        return np.arccos(np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B)))

    # y = Ax + b || Ax - y + b
    @staticmethod
    def isPointBetween2Lane(leftLaneA, leftLaneB, rightLaneA, rightLaneB, point):
        if point[1] > leftLaneA*point[0] + leftLaneB and point[1] > rightLaneA*point[0] + rightLaneB:
            return True
        return False

    @staticmethod
    def publicVelocity(straight, angular):
        # myTwist = Twist()
        # myTwist.linear.x = straight
        # myTwist.angular.z = angular
        # pub.publish(myTwist)
        msg = json.dumps({"linear": straight, "angular": angular})
        pub.publish(msg)


class CombineLidarLane:
    def __init__(self) -> None:
        self.lane_signal_subscriber = rospy.Subscriber(
            TOPIC_NAME_LANE_DETECTION, String, self.updateLaneDetectionSignal
        )
        self.lidar_signal_subscriber = rospy.Subscriber(
            TOPIC_NAME_LIDAR, LaserScan, self.updateLidarSignal)
        self.ranges = [1 for _ in range(360)]
        self.tracePath = []
        self.straightVel = 0
        self.turnVel = 0

        # Goal of image
        self.goalX = WIDTH_SIMULATE_MAP//2
        self.goalY = 0

        # Position of lane
        self.lastTimeReciveLane = time.time()
        self.leftBottomLaneX = 0
        self.leftBottomLaneY = HEIGH_SIMULATE_MAP - 1

        self.leftTopLaneX = WIDTH_SIMULATE_MAP//4
        self.leftTopLaneY = 0

        self.rightTopLaneX = WIDTH_SIMULATE_MAP - self.leftTopLaneX
        self.rightTopLaneY = 0

        self.rightBottomLaneX = WIDTH_SIMULATE_MAP - 1
        self.rightBottomLaneY = HEIGH_SIMULATE_MAP - 1

        # Pt duong thang y = Ax + b     =>      Ax - y + b = 0
        self.leftLaneA = (self.leftTopLaneY - self.leftBottomLaneY) / \
            (self.leftTopLaneX - self.leftBottomLaneX)
        self.leftLaneB = self.leftTopLaneY - self.leftLaneA*self.leftTopLaneX

        self.rightLaneA = (self.rightTopLaneY - self.rightBottomLaneY) / \
            (self.rightTopLaneX - self.rightBottomLaneX)
        self.rightLaneB = self.rightTopLaneY - self.rightLaneA * self.rightTopLaneX


    def drawLanesOnMap(self, simMap):
        color = 255
        thickness = 5
        simMap = cv2.line(simMap, (self.leftBottomLaneX, self.leftBottomLaneY),
                          (self.leftTopLaneX, self.leftTopLaneY), color, thickness)
        simMap = cv2.line(simMap, (self.rightBottomLaneX, self.rightBottomLaneY),
                          (self.rightTopLaneX, self.rightTopLaneY), color, thickness)
        return simMap

    def updateLaneDetectionSignal(self, msg):
        self.lastTimeReciveLane = time.time()
        # parsed = json.loads(msg.data)
        # self.goalX = parsed["something-here"]
        # self.goalY = parsed["something-here"]
        # self.leftBottomLaneX = parsed["something-here"]
        # self.leftBottomLaneY = parsed["something-here"]
        # self.leftTopLaneX = parsed["something-here"]
        # self.leftTopLaneY = parsed["something-here"]
        # self.rightTopLaneX = parsed["something-here"]
        # self.rightTopLaneY = parsed["something-here"]
        # self.rightBottomLaneX = parsed["something-here"]
        # self.rightBottomLaneY = parsed["something-here"]

        self.goalX = WIDTH_SIMULATE_MAP//2
        self.goalY = 0

    def updateLidarSignal(self, scan):
        # Front of robot index = 0, anti clockwise +1 index (left = 90 deg)
        self.ranges = scan.ranges

        # self.updateVelocity()

    def updateVelocity(self, event):
        # self.getPath()
        Utils.publicVelocity(self.straightVel, self.turnVel)
    
    def rotatePoint(self):
        deltaTime = time.time() - self.lastTimeReciveLane
        self.lastTimeReciveLane = time.time()
        R = self.straightVel / self.turnVel
        isRight = self.turnVel < 0
        alpha = abs(self.turnVel * deltaTime)
        curPos = [HEIGH_SIMULATE_MAP, int(WIDTH_SIMULATE_MAP / 2)]
        newPos = [0, 0]
        if isRight:
            newPos = [curPos[0] + R + R*math.cos(alpha), curPos[1] + R*math.sin(alpha)]
        else:
            newPos = [curPos[0] - R - R*math.cos(alpha), curPos[1] + R*math.sin(alpha)]
        vector = Utils.getVectorAB(curPos, newPos)
        invertVec = Utils.getInvertVector(vector)
        pass

    def pathFinding(self, event):
        # Convert lidar distance signal, result represent left to right (0 -> 179)
        angleList = np.arange(start=0, stop=180, step=1, dtype=np.int16)
        angleList = angleList * np.pi/180.
        convertedLidarSignalBaseAngle = np.zeros(shape=((180, )))

        # 0 -> 180 in anti clockwise
        convertedLidarSignalBaseAngle[0:90] = self.ranges[270:360]
        convertedLidarSignalBaseAngle[90:180] = self.ranges[0:90]

        # Convert distance signal into vertical axis, then find the scale factor
        scaleFactor = HEIGH_SIMULATE_MAP/LIDAR_MAX_RANGE

        # Projected onto the Y axis
        scaledLidarSignalBaseAngle = convertedLidarSignalBaseAngle * \
            scaleFactor*np.sin(angleList)

        # Finding the scale distance param to display on image
        simulateMap = np.zeros(
            shape=(HEIGH_SIMULATE_MAP, WIDTH_SIMULATE_MAP), dtype='uint8')
        pathOnlyMap = np.zeros(
            shape=(HEIGH_SIMULATE_MAP, WIDTH_SIMULATE_MAP), dtype='uint8')
        coordinateYObstacleSimulationMap = HEIGH_SIMULATE_MAP - \
            scaledLidarSignalBaseAngle.astype(np.int16)
        coordinateXObstacleSimulationMap = WIDTH_SIMULATE_MAP//2 + \
            ((scaledLidarSignalBaseAngle/(np.sin(angleList) + 0.0001))
             * (np.cos(angleList))).astype(np.int16)
        filteredIndex = np.where(((coordinateYObstacleSimulationMap >= 0) & (coordinateYObstacleSimulationMap < HEIGH_SIMULATE_MAP) & (
            coordinateXObstacleSimulationMap >= 0) & (coordinateXObstacleSimulationMap < WIDTH_SIMULATE_MAP)))

        # Make obstacle bigger
        simulateMap[coordinateYObstacleSimulationMap[filteredIndex],
                    coordinateXObstacleSimulationMap[filteredIndex]] = BLOCKED_COLOR
        # Make obstacle bigger - Option 1
        # for y, x in zip(coordinateYObstacleSimulationMap[filteredIndex], coordinateXObstacleSimulationMap[filteredIndex]):
        #     simulateMap[max(0, y - DELTA_Y) : min(HEIGH_SIMULATE_MAP, y + DELTA_Y), max(0, x - DELTA_X) : min(WIDTH_SIMULATE_MAP, x + DELTA_X)] = BLOCKED_COLOR
        # Displayed on the pre-pathplanning image

        # Make obstacle bigger - Option 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DELTA, DELTA))
        simulateMap = cv2.dilate(simulateMap, kernel)

        # Draw path from lane detection in to map
        simulateMap = self.drawLanesOnMap(simulateMap)

        # Path finding here
        initalX, initialY = WIDTH_SIMULATE_MAP//2, HEIGH_SIMULATE_MAP - 1

        # If not exist lane to go -> do nothing
        rowAllBlocked = np.all(simulateMap == BLOCKED_COLOR, axis=1)
        isExistLaneToGo = not np.any(rowAllBlocked == True)

        # Get Path
        t1 = time.time()
        self.tracePath = []

        if (isExistLaneToGo):
            nonBlockedListCoor = [np.where(row != BLOCKED_COLOR)[
                0] for row in simulateMap]
            nonBlockedListCoor = np.array(nonBlockedListCoor, dtype=object)
            for row in range(1, HEIGH_SIMULATE_MAP - 1):
                minIndexDiffX = np.argmin(
                    np.abs(nonBlockedListCoor[row] - self.goalX))
                self.tracePath.append(
                    [nonBlockedListCoor[row][minIndexDiffX], row])

        t2 = time.time()
        print(t2 - t1)
        if len(self.tracePath):
            for coor in self.tracePath:
                x, y = coor[0], coor[1]
                # simulateMap =  cv2.circle(simulateMap, (x, y), 1, (255, 0, 0), 1)
                pathOnlyMap[y, x] = 255
                simulateMap[y, x] = 255

        cv2.imshow("simulate map", simulateMap)
        cv2.imshow("path only map", pathOnlyMap)
        self.getVelocity()
        if cv2.waitKey(1) == ord('q'):
            return

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
        point15th = len(self.tracePath) - 1 - NUM_POINTS_OF_DIRECTION
        if (point15th < 0):
            return
    
        # check target point is in lane
        polygon = Polygon([
            (self.leftBottomLaneX, self.leftBottomLaneY),
            (self.rightBottomLaneX, self.rightBottomLaneY),
            (self.rightTopLaneX, self.rightTopLaneY),
            (self.leftTopLaneX, self.leftTopLaneY)
        ])
        if not polygon.contains(Point(self.tracePath[point15th])):
            self.straightVel = 0
            self.turnVel = 0
            return
        
        vecDirection = Utils.getVectorAB([HEIGH_SIMULATE_MAP, int(
            WIDTH_SIMULATE_MAP / 2)], self.tracePath[point15th])
        vecZero = (1, 0)
        angle = Utils.getAngleOfVectors(vecDirection, vecZero)
        isRight = angle < math.radians(90)
        alpha = abs(angle - math.radians(90))
        straightVel = math.cos(alpha) * MAX_STRAIGHT_VELOCITY
        turnVel = math.sin(alpha) * MAX_TURN_VELOCITY


        if isRight:
            self.straightVel = straightVel
            self.turnVel = -turnVel
            # self.updateVelocity()
            return

        self.straightVel = straightVel
        self.turnVel = turnVel


if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_AVOIDANCE, anonymous=True)
        avoidance = CombineLidarLane()
        rospy.Timer(rospy.Duration(0.2), avoidance.pathFinding)  # 0.05
        rospy.Timer(rospy.Duration(0.1), avoidance.updateVelocity)  # 0.01

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
