#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import numpy as np
import json
import time

# NODE_NAME_AVOIDANCE = rospy.get_param('NODE_NAME_AVOIDANCE')
NODE_NAME_DECISION_MAKING = rospy.get_param('NODE_NAME_DECISION_MAKING')

TOPIC_TRAFFIC_SIGN = rospy.get_param('TOPIC_NAME_TRAFFIC_SIGN')
TOPIC_TRAFFIC_LIGHT = rospy.get_param('TOPIC_NAME_TRAFFIC_LIGHT')
TOPIC_AVOIDANCE = rospy.get_param('TOPIC_NAME_AVOIDANCE')
TOPIC_VELOCITY = rospy.get_param('TOPIC_NAME_VELOCITY')

RESPONSE_LIGTH = rospy.get_param('RESPONSE_LIGTH')
RESPONSE_SIGN = rospy.get_param('RESPONSE_SIGN')


# Hardcode turn left and right
STRAIGHT_VEL = 0.2
STRAIGHT_TIME = 4
TURN_VEL = 0.1
TURN_TIME = 5
## total time to turn is SRTAIGHT_TIME + TURN_TIME + SRTAIGHT_TIME
TOTAL_TIME = 2*STRAIGHT_TIME + TURN_TIME

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
        self.veloPublisher = rospy.Publisher(TOPIC_VELOCITY, Twist, queue_size=1)

        self.linear = 0
        self.angular = 0
        
        self.backupLinear = 0
        self.backupAngular = 0
        
        # PRIOTIRY SIGNAL
        self.light = RESPONSE_LIGTH['NONE']
        self.sign = RESPONSE_SIGN['NONE']
        self.isInProcess = False
        self.action = None
        self.processStartTime = 0
        self.processTime = 0
        self.processStep = 0    # 0: straight, 1: turn, 2: straight
    
    # for case that don't care AVOIDANCE_MODULE velocity
    def isExistPriority(self):
        return self.light is RESPONSE_LIGTH['RED'] or self.light is RESPONSE_LIGTH['YELLOW'] or self.sign is RESPONSE_SIGN['STOP']
    
    def trafficLightCallback(self, colorMsg):
        self.light = colorMsg
        if colorMsg is RESPONSE_LIGTH['RED']:
            self.linear = 0
            self.angular = 0
        elif colorMsg is RESPONSE_LIGTH['YELLOW']:
            self.linear = self.linear // 2
            self.angular = self.angular // 2
        # green and none case
        else:
            self.linear = self.backupLinear
            self.angular = self.backupAngular
            
    def trafficSignCallback(self, signMsg):
        # parsed = json.loads(msg.data)
        # if parsed['type'] == 'STOP':
        self.sign = signMsg
        if signMsg is RESPONSE_SIGN['STOP']:
            self.linear = 0
            self.angular = 0
        elif signMsg is RESPONSE_SIGN['LEFT'] or signMsg is RESPONSE_SIGN['RIGHT']:
            self.isInProcess = True
            self.action = signMsg
        # forward and none case (and forbid case)
        else:
            self.linear = self.backupLinear
            self.angular = self.backupAngular


    def avoidanceCallback(self, msg):
        parsed = json.loads(msg.data)
        self.backupLinear = parsed['linear']
        self.backupAngular = parsed['angular']
        if self.isExistPriority():
            return
        else:
            self.linear = parsed['linear']
            self.angular = parsed['angular']
    
    def publishVelo(self, event):
        myTwist = Twist()
        
        # check hardcode turn
        if self.isInProcess:
            # first time
            if self.processStartTime is 0:
                self.processStartTime = time.time()
            else:
                # update time
                self.processTime = time.time() - self.processStartTime
                
                # update step
                if 0 <= self.processTime < STRAIGHT_TIME:
                    self.processStep = 0
                elif STRAIGHT_TIME <= self.processTime < STRAIGHT_TIME + TURN_TIME:
                    self.processStep = 1
                elif STRAIGHT_TIME + TURN_TIME <= self.processTime < TOTAL_TIME:
                    self.processStep = 2
            
            
            # check done condition
            if self.processStep is 2 and self.processTime > TOTAL_TIME:
                self.isInProcess = False
                self.action = None
                return self.veloPublisher.publish(myTwist)
            
            if self.processStep is 0 or self.processStep is 2:
                # go Straight at step 0 and 2
                myTwist.linear.x = STRAIGHT_VEL
                myTwist.angular.z = 0
            # turn at step 1
            elif self.action is RESPONSE_SIGN['LEFT']:
                self.linear = 0
                self.angular = TURN_VEL
            elif self.action is RESPONSE_SIGN['RIGHT']:
                self.linear = 0
                self.angular = -TURN_VEL
            return self.veloPublisher.publish(myTwist)
        
        # normal flow
        myTwist.linear.x = self.linear
        myTwist.angular.z = self.angular
        self.veloPublisher.publish(myTwist)
        
if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME_DECISION_MAKING, anonymous=True)
        decisionMaking = CombineDecisionModule()
        rospy.Timer(rospy.Duration(0.1), decisionMaking.publishVelo) # 0.01
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
