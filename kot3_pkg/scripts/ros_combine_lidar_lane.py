#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import importlib
import json
import numpy as np
import cv2

# Test here
from constant import *

TOPIC_NAME_AVOIDANCE = 'avoidance_topic'
NODE_NAME_AVOIDANCE = 'avoidance_node_name'
TOPIC_NAME_LIDAR = '/scan' # Check it
LIDAR_MAX_RANGE = 3 # metters, unit
WIDTH_SIMULATE_MAP = 2*LIDAR_MAX_RANGE*100
HEIGH_SIMULATE_MAP = LIDAR_MAX_RANGE*100

class CombineLidarLane:
    def __init__(self) -> None:
        # Subscribe from "AI Lane Dection" module
        # self.lane_signal_subscriber = rospy.Subscriber(
        #     TOPIC_NAME_LANE_DETECTION, String, self.updateEnvironmentLaneSignal)
        self.lidar_signal_subscriber = rospy.Subscriber(
            TOPIC_NAME_LIDAR, LaserScan, self.updateLidarSignal)

        # Publisher
        # self.avoidance_publisher = rospy.Publisher(
        #     TOPIC_NAME_AVOIDANCE, String, queue_size=1)

    def updateLidarSignal(self, scan):
        print("test", NODE_NAME_TRAFFIC_LIGHTS)
        # Front of robot index = 0, anti clockwise +1 index (left = 90 deg)
        ranges = scan.ranges
        # intensities = scan.intensities

        # Convert lidar distance signal, result represent left to right (0 -> 179)
        angleList = np.arange(start=0, stop=180, step=1, dtype=np.int16)
        angleList = angleList * np.pi/180.
        convertedLidarSignalBaseAngle = np.zeros(shape=((180, )))

        # 0 -> 180 in anti clockwise
        # convertedLidarSignalBaseAngle[0:90] = ranges[0:90][::-1]
        # convertedLidarSignalBaseAngle[90:180] = ranges[270:360][::-1]
        convertedLidarSignalBaseAngle[0:90] = ranges[270:360]
        convertedLidarSignalBaseAngle[90:180] = ranges[0:90]
        
        # Convert distance signal into vertical axis, then find the scale factor
        scaleFactor = HEIGH_SIMULATE_MAP/LIDAR_MAX_RANGE

        # Projected onto the Y axis
        scaledLidarSignalBaseAngle = convertedLidarSignalBaseAngle*scaleFactor*np.sin(angleList)

        # Finding the scale distance param to display on image
        simulateMap = np.zeros(shape=(HEIGH_SIMULATE_MAP, WIDTH_SIMULATE_MAP), dtype='uint8')
        coordinateYObstacleSimulationMap = HEIGH_SIMULATE_MAP - scaledLidarSignalBaseAngle.astype(np.int16) 
        coordinateXObstacleSimulationMap = WIDTH_SIMULATE_MAP//2 + ((scaledLidarSignalBaseAngle/(np.sin(angleList) + 0.0001))*(np.cos(angleList))).astype(np.int16)
        

        filteredIndex = np.where((coordinateYObstacleSimulationMap >= 0) & (coordinateYObstacleSimulationMap < HEIGH_SIMULATE_MAP) & (coordinateXObstacleSimulationMap >= 0) & (coordinateXObstacleSimulationMap < WIDTH_SIMULATE_MAP))
        
        # Make obstacle bigger
        simulateMap[coordinateYObstacleSimulationMap[filteredIndex], coordinateXObstacleSimulationMap[filteredIndex]] = 255
        cv2.imshow("aaaa", simulateMap)
        if cv2.waitKey(1) == ord('q'):
            return
        # print(simulateMap)
        # Displayed on the pre-pathplanning image

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


if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_AVOIDANCE, anonymous=True)
        avoidance = CombineLidarLane()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
