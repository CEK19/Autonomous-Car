#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import importlib
import json
import numpy as np
import cv2
import heapq
import time
import math

# Test here

TOPIC_NAME_AVOIDANCE = 'avoidance_topic'
NODE_NAME_AVOIDANCE = 'avoidance_node_name'
TOPIC_NAME_LIDAR = '/scan' # Check it
LIDAR_MAX_RANGE = 3 # metters, unit
WIDTH_SIMULATE_MAP = 300
HEIGH_SIMULATE_MAP = 150
BLOCKED_COLOR = 255
DELTA_X = 4
DELTA_Y = 4
NUM_POINTS_OF_DIRECTION = 15
MAX_STRAIGHT_VELOCITY = 10
MAX_TURN_VELOCITY = 4

class Utils:
    @staticmethod
    def getVectorAB(A, B):
        return B[0] - A[0], B[1] - A[1]
    
    @staticmethod
    def getAngleOfVectors(A, B):
        return np.arccos(np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B)))
    

class CombineLidarLane:
    def __init__(self) -> None:
        # Subscribe from "AI Lane Dection" module
        # self.lane_signal_subscriber = rospy.Subscriber(
        #     TOPIC_NAME_LANE_DETECTION, String, self.updateEnvironmentLaneSignal)
        self.lidar_signal_subscriber = rospy.Subscriber(
            TOPIC_NAME_LIDAR, LaserScan, self.updateLidarSignal)
        self.ranges = [1 for _ in range(360)]
        self.tracePath = np.empty(shape=(0, 2), dtype=int)

        # Publisher
        # self.avoidance_publisher = rospy.Publisher(
        #     TOPIC_NAME_AVOIDANCE, String, queue_size=1)

    def updateLidarSignal(self, scan):
        # Front of robot index = 0, anti clockwise +1 index (left = 90 deg)
        self.ranges = scan.ranges
                
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
        scaledLidarSignalBaseAngle = convertedLidarSignalBaseAngle*scaleFactor*np.sin(angleList)

        # Finding the scale distance param to display on image
        simulateMap = np.zeros(shape=(HEIGH_SIMULATE_MAP, WIDTH_SIMULATE_MAP), dtype='uint8')
        pathOnlyMap = np.zeros(shape=(HEIGH_SIMULATE_MAP, WIDTH_SIMULATE_MAP), dtype='uint8')
        coordinateYObstacleSimulationMap = HEIGH_SIMULATE_MAP - scaledLidarSignalBaseAngle.astype(np.int16) 
        coordinateXObstacleSimulationMap = WIDTH_SIMULATE_MAP//2 + ((scaledLidarSignalBaseAngle/(np.sin(angleList) + 0.0001))*(np.cos(angleList))).astype(np.int16)        
        filteredIndex = np.where(((coordinateYObstacleSimulationMap >= 0) & (coordinateYObstacleSimulationMap < HEIGH_SIMULATE_MAP) & (coordinateXObstacleSimulationMap >= 0) & (coordinateXObstacleSimulationMap < WIDTH_SIMULATE_MAP)))
        
        # Make obstacle bigger
        simulateMap[coordinateYObstacleSimulationMap[filteredIndex], coordinateXObstacleSimulationMap[filteredIndex]] = BLOCKED_COLOR
        for y, x in zip(coordinateYObstacleSimulationMap[filteredIndex], coordinateXObstacleSimulationMap[filteredIndex]):
            simulateMap[max(0, y - DELTA_Y) : min(HEIGH_SIMULATE_MAP, y + DELTA_Y), max(0, x - DELTA_X) : min(WIDTH_SIMULATE_MAP, x + DELTA_X)] = BLOCKED_COLOR
        # Displayed on the pre-pathplanning image

        # Path finding here
        goalX, goalY = WIDTH_SIMULATE_MAP//2, 0
        initalX, initialY = WIDTH_SIMULATE_MAP//2, HEIGH_SIMULATE_MAP - 1

        # If not exist lane to go -> do nothing
        rowAllBlocked = np.all(simulateMap == BLOCKED_COLOR, axis=1)
        isExistLaneToGo = not np.any(rowAllBlocked == True)

        # Get Path
        t1 = time.time()
        # aStar = AStar(matrix=simulateMap, start=(initalX, initialY), goal=(goalX, goalY))
        # tracePath = aStar.find_path()
        self.tracePath = np.empty(shape=(0, 2), dtype=int)
        if isExistLaneToGo:
            for row in range(1, HEIGH_SIMULATE_MAP - 1):
                nonBlockedListCoor = np.empty(shape=(0, 2), dtype=int)
                for col in range(0, WIDTH_SIMULATE_MAP):
                    if (simulateMap[row, col] != BLOCKED_COLOR):
                        nonBlockedListCoor = np.append(nonBlockedListCoor, [[col, row]], axis=0)

                tmpIndex = np.argmin(np.abs(nonBlockedListCoor[:, 0] - goalX))
                self.tracePath = np.append(self.tracePath, [nonBlockedListCoor[tmpIndex]], axis=0)

        t2 = time.time()
        # print(t2 - t1)
        # print(tracePath)
        if len(self.tracePath):
            for coor in self.tracePath:
                x, y = coor[0], coor[1]
                # simulateMap =  cv2.circle(simulateMap, (x, y), 1, (255, 0, 0), 1)
                pathOnlyMap[y, x] = 255
                simulateMap[y, x] = 255

        cv2.imshow("simulate map", simulateMap)
        cv2.imshow("path only map", pathOnlyMap)
        if cv2.waitKey(1) == ord('q'):
            return   

    def sendActionToTopic(self, action):
        """
        Input: action
        Ouput: None
        """
        message = json.dumps({"action": action})
        self.avoidance_publisher.publish(message)

    def solve(self, data):
        self.sendActionToTopic(action)
        rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    
    def getVelocity(self):
        vecDirection = Utils.getVectorAB(self.tracePath[0], self.tracePath[NUM_POINTS_OF_DIRECTION])
        vecZero = (1, 0)
        angle = Utils.getAngleOfVectors(vecDirection, vecZero)
        isRight = angle < math.radians(90)
        # isCenter = math.radians(87) < angle < math.radians(93)
        alpha = abs(angle - math.radians(90))
        straightVel = math.cos(alpha) * MAX_STRAIGHT_VELOCITY
        turnVel = math.sin(alpha) * MAX_TURN_VELOCITY
        if isRight:
            return straightVel, turnVel
        
        return straightVel, -turnVel


if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_AVOIDANCE, anonymous=True)
        avoidance = CombineLidarLane()
        rospy.Timer(rospy.Duration(0.25), avoidance.pathFinding)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
