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

        # Publisher
        self.control_velocity_publisher = rospy.Publisher(TOPIC_NAME_VELOCITY, String, queue_size=1)

        # Data update
        self.traffictLightSignal = dict()
        self.trafficSignsSignal = dict()
        self.avoidanceActionSignal = dict()

        # check update param
        self.isUpdatedTraffictLight = False
        self.isUpdatedTrafficSign = False
        self.isUpdatedActionFromAvoidanceModules = False

    def isEnoughDataToMakeDecision(self):
        return self.isUpdatedTraffictLight and self.isUpdatedTrafficSign and self.isUpdatedActionFromAvoidanceModules

    def clearData(self):
        self.isUpdatedTraffictLight = False
        self.isUpdatedTrafficSign = False
        self.isUpdatedActionFromAvoidanceModules = False

    def clearData(self):
        self.isUpdatedTraffictLight = False
        self.isUpdatedTrafficSign = False
        self.isUpdatedActionFromAvoidanceModules = False

    def updateTrafficLightSignal(self, data):
        # Do preprocessing data here ...
        self.traffictLightSignal = data.data
        self.isUpdatedTraffictLight = True
        if self.isEnoughDataToMakeDecision():
            self.makeActionToRobot()

    def updateTrafficSignsSignal(self, data):
        # Do preprocessing data here ...
        self.trafficSignsSignal = data.data
        self.isUpdatedTrafficSign = True
        if self.isEnoughDataToMakeDecision():
            self.makeActionToRobot()

    def updateActionFromAvoidanceModuleSignal(self, data):
        # Do preprocessing data here ...
        self.avoidanceActionSignal = data.data
        self,isUpdatedActionFromAvoidanceModules = True
        if self.isEnoughDataToMakeDecision():
            self.makeActionToRobot()

    def makeActionToRobot(self):
        # Do bla bla bla
        #...
        self.control_velocity_publisher.publish("some data here")
        

if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_DECISION_MAKING, anonymous=True)
        decisionMaking = CombineDecisionModule()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
