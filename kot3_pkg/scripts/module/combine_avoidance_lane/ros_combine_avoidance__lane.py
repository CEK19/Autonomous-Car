#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from constant import *
import json
import numpy as np


class AvoidanceModule:
    def __init__(self) -> None:
        # Subscribe from "AI Lane Dection" module
        # self.lane_signal_subscriber = rospy.Subscriber(
        #     TOPIC_NAME_LANE_DETECTION, String, self.updateEnvironmentLaneSignal)
        self.lidar_signal_subscriber = rospy.Subscriber(
            TOPIC_NAME_LIDAR, LaserScan, self.updateEnvironmentLidarSignal)

        # Publisher
        # self.avoidance_publisher = rospy.Publisher(
        #     TOPIC_NAME_AVOIDANCE, String, queue_size=1)


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
        avoidance = AvoidanceModule()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
