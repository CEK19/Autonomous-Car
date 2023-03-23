#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import numpy as np
import json

NODE_NAME_AVOIDANCE = rospy.get_param('NODE_NAME_AVOIDANCE')
TOPIC_TRAFFIC_SIGN = rospy.get_param('TOPIC_NAME_TRAFFIC_SIGN')
TOPIC_AVOIDANCE = rospy.get_param('TOPIC_NAME_AVOIDANCE')
TOPIC_TRAFFIC_LIGHT = rospy.get_param('TOPIC_NAME_TRAFFIC_LIGHT')
TOPIC_VELOCITY = rospy.get_param('TOPIC_NAME_VELOCITY')
NODE_NAME_DECISION_MAKING = rospy.get_param('NODE_NAME_DECISION_MAKING')

class CombineDecisionModule:

    def __init__(self):
        
        # Subscriber
        self.trafficLightSubscriber = rospy.Subscriber(
            TOPIC_TRAFFIC_LIGHT, String, self.trafficLightCallback)
        self.trafficSignSubscriber = rospy.Subscriber(
            TOPIC_TRAFFIC_SIGN, String, self.trafficSignCallback)
        self.avoidanceSubscriber = rospy.Subscriber(
            TOPIC_AVOIDANCE, String, self.avoidanceCallback)
        
        # Publisher
        self.veloPublisher = rospy.Publisher(TOPIC_VELOCITY, Twist, queue_size=10)

        self.linear = 0
        self.angular = 0
        
        self.backupLinear = 0
        self.backupAngular = 0
        
        # PRIOTIRY SIGNAL
        self.RED_LIGHT = False
        self.STOP_SIGN = False
    
    def isExistPriority(self):
        return self.RED_LIGHT or self.STOP_SIGN
    
    def trafficLightCallback(self, msg):
        parsed = json.loads(msg.data)
        if parsed['color'] == 'RED':
            self.RED_LIGHT = True
            self.linear = 0
            self.angular = 0
        else:
            self.RED_LIGHT = False
            self.linear = self.backupLinear
            self.angular = self.backupAngular
    def trafficSignCallback(self, msg):
        parsed = json.loads(msg.data)
        if parsed['type'] == 'STOP':
            self.STOP_SIGN = True
            self.linear = 0
            self.angular = 0
        else:
            self.STOP_SIGN = False
    def avoidanceCallback(self, msg):
        parsed = json.loads(msg.data)
        self.backupLinear = parsed['linear']
        self.backupAngular = parsed['angular']
        if not self.isExistPriority():
            self.linear = parsed['linear']
            self.angular = parsed['angular']
    
    def publishVelo(self):
        myTwist = Twist()
        myTwist.linear.x = self.linear
        myTwist.angular.z = self.angular
        self.veloPublisher.publish(myTwist)
        
if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_DECISION_MAKING, anonymous=True)
        decisionMaking = CombineDecisionModule()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
