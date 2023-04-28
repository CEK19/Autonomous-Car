#!/usr/bin/env python

import time
import rospy
from geometry_msgs.msg import Twist

TOPIC_NAME_VELOCITY = rospy.get_param('TOPIC_NAME_VELOCITY')
count = 0
startTime = 0
totalTime = 0

def callback(data):
    global count
    count = count + 1
    totalTime = time.time() - startTime
    if totalTime is not 0:
        print("fps: ", count/totalTime)


def listener():
    rospy.init_node('listener', anonymous=True)

    # rospy.Subscriber('chatter', String, callback)
    rospy.Subscriber('/cmd_vel', Twist, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    startTime = time.time()
    listener()
