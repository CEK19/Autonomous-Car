#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from constant import *
import json
import numpy as np


class AvoidanceModule:
    def __init__(self) -> None:

        # Subcriber
        self.lane_signal_subscriber = rospy.Subscriber(
            TOPIC_NAME_LANE_DETECTION, String, self.updateEnvironmentLaneSignal)
        self.lidar_signal_subscriber = rospy.Subscriber(
            TOPIC_NAME_LIDAR, LaserScan, self.updateEnvironmentLidarSignal)
        self.control_action_subscriber = rospy.Subscriber(
            TOPIC_NAME_ACTION_DECISION, String, self.updateEnvironmentControlActionSignal)

        # Publisher
        self.avoidance_publisher = rospy.Publisher(
            TOPIC_NAME_AVOIDANCE, String, queue_size=1)

        # Table for RL
        self.prebuiltQTable = self.loadQTableFromFile()

        # Data collection
        self.rawLidarSignal = []  # Data from lidar
        self.ratioLeft = 0.5  # Data from lane detection module
        self.alpha = 0  # Alpha

        self.isUpdatedRawLidarSignal = False
        self.isUpdatedLaneSignal = False
        self.isUpdatedActionSignal = False

    def isEnoughDataToMakeDecision(self):
        return self.isUpdatedLaneSignal and self.isUpdatedLaneSignal and self.isUpdatedActionSignal

    def loadQTableFromFile(self):
        """
        Input: File
        Ouput: data from file
        """
        file = open(ASSETS.Q_TABLE_DATA, "r")
        RLInFile = file.read()
        if not RLInFile:
            raise Exception(EXCEPTION.NO_Q_TABLE)
        else:
            return json.loads(RLInFile)

    def updateEnvironmentLidarSignal(self, data):
        self.rawLidarSignal = data.data
        self.isUpdatedRawLidarSignal = True
        print("from lidar module", data)
        if self.isEnoughDataToMakeDecision():
            self.solve()

    def updateEnvironmentLaneSignal(self, data):
        self.ratioLeft = data.data
        self.isUpdatedLaneSignal = True
        print("from lane module", data)
        if self.isEnoughDataToMakeDecision():
            self.solve()

    def updateEnvironmentControlActionSignal(self, data):
        self.alpha = data.data
        self.isUpdatedActionSignal = True
        print("from lane control signal", data)
        if self.isEnoughDataToMakeDecision():
            self.solve()

    def convertSignalToState(self, updatedSignal):
        """
        Input: ratio_left, lidar, alpha
        Ouput: ABCDE
            A (0-1): Vật thể vùng 1
            B (0-1): Vật thể vùng 2
            C (0-1): Vật thể vùng 3
            D(0-9): Hướng di chuyển của robot, vì robot chỉ quay một góc 18 độ mỗi lần, nên trong phạm vi góc -90 đến 90, ta có 10 giá trị góc mà robot có thể tồn tại.
            E(0-9): Đại diện cho vị trí robot so với chiều ngang của vùng xanh lá. Ta chia vùng xanh lá ra thành 10 vùng theo chiều dọc và đánh số từ 0 đến 9 từ trái qua phải
            Tổng cộng, ta sẽ có tất cả 800 state và 4 action
        """
        pass

    def decideActionBaseOnCurrentState(self, action):  # Done
        """
        Input: ABCDE
        Ouput: action
        """
        return np.argmax(self.prebuiltQTable[action])

    def sendActionToTopic(self, action):
        """
        Input: action
        Ouput: None
        """
        message = json.dumps({"action": action})
        self.avoidance_publisher.publish(message)

    def solve(self, data):
        updatedSignal = self.updateEnvironmentSignal()
        state = self.convertSignalToState(updatedSignal)
        action = self.decideActionBaseOnCurrentState(state)
        self.sendActionToTopic(action)
        rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)


if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_AVOIDANCE, anonymous=True)
        avoidance = AvoidanceModule()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
