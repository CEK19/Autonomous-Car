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

# Test here
NODE_NAME_TRAFFIC_LIGHT = rospy.get_param('NODE_NAME_TRAFFIC_LIGHT')

TOPIC_NAME_VELOCITY = rospy.get_param('TOPIC_NAME_VELOCITY')
TOPIC_NAME_LIDAR = rospy.get_param('TOPIC_NAME_LIDAR')
TOPIC_NAME_AVOIDANCE = rospy.get_param('TOPIC_NAME_AVOIDANCE')

LIDAR_MAX_RANGE = 3 # metters, unit
WIDTH_SIMULATE_MAP = 2*LIDAR_MAX_RANGE*100
HEIGH_SIMULATE_MAP = LIDAR_MAX_RANGE*100
BLOCKED_COLOR = 255
DELTA = 50 
DELTA_X = DELTA
DELTA_Y = DELTA
NUM_POINTS_OF_DIRECTION = 35 # 27
MAX_STRAIGHT_VELOCITY = 0.2  # 0.2
MAX_TURN_VELOCITY = 2.0  # 1.4


pub = rospy.Publisher(TOPIC_NAME_AVOIDANCE, String, queue_size=1)

class Utils:
    @staticmethod
    def getVectorAB(A, B):
        return B[0] - A[0], B[1] - A[1]
    
    @staticmethod
    def getAngleOfVectors(A, B):
        return np.arccos(np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B)))

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
        # Subscribe from "AI Lane Dection" module
        # self.lane_signal_subscriber = rospy.Subscriber(
        #     TOPIC_NAME_LANE_DETECTION, String, self.updateEnvironmentLaneSignal)
        self.lidar_signal_subscriber = rospy.Subscriber(
            TOPIC_NAME_LIDAR, LaserScan, self.updateLidarSignal)
        self.ranges = [1 for _ in range(360)]
        self.tracePath = []
        self.straightVel = 0
        self.turnVel = 0


if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_TRAFFIC_LIGHT, anonymous=True)
        avoidance = CombineLidarLane()
        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass