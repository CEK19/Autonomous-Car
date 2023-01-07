#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from constant import *
import numpy as np


class CombineDecisionModule:

    def __init__(self) -> None:

        # Subcriber
        self.traffic_lights_subscriber = rospy.Subscriber(
            TOPIC_NAME__TRAFFIC_LIGHTS, String, self.updateTrafficLightSignal)

        self.traffic_signs_subsciber = rospy.Subscriber(
            TOPIC_NAME_TRAFFIC_SIGNS, String, self.updateTrafficSignsSignal)

        self.avoidance_module_subscriber = rospy.Subscriber(
            TOPIC_NAME_AVOIDANCE, String, self.updateActionFromAvoidanceModuleSignal
        )

        # Data update
        self.traffictLightSignal = dict()
        self.trafficSignsSignal = dict()
        self.avoidanceActionSignal = dict()

        # check update param
        self.isUpdatedTraffictLight = False
        self.isUpdatedTrafficSight = False
        self.isUpdatedActionFromAvoidanceModules = False

    def isEnoughDataToMakeDecision(self):
        return self.isUpdatedTraffictLight and self.isUpdatedTrafficSight and self.isUpdatedActionFromAvoidanceModules

    def clearData(self):
        self.isUpdatedTraffictLight = False
        self.isUpdatedTrafficSight = False
        self.isUpdatedActionFromAvoidanceModules = False

    def updateTrafficLightSignal(self, data):
        self.isUpdatedTraffictLight = data.data
        self.isUpdatedRawLidarSignal = True
        if self.isEnoughDataToMakeDecision():
            self.makeActionToRobot()

    def updateTrafficSignsSignal(self, data):
        pass

    def updateActionFromAvoidanceModuleSignal(self, data):
        pass

    def makeActionToRobot():
        pass

if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_DECISION_MAKING, anonymous=True)
        decisionMaking = CombineDecisionModule()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
