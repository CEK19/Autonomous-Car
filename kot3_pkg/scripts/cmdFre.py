#!/usr/bin/env python

import time
import rospy
from geometry_msgs.msg import Twist

TOPIC_NAME_VELOCITY = rospy.get_param('TOPIC_NAME_VELOCITY')
isFirst = True
count = 0
startTime = 0

def callback(data):
    global count, isFirst, startTime
    if (isFirst):
        startTime = time.time()
        isFirst = False
    count = count + 1
    totalTime = time.time() - startTime
    if totalTime != 0:
        print("fps: ", count/totalTime, "time: ", totalTime)


def listener():
    rospy.init_node('listener', anonymous=True)

    # rospy.Subscriber('chatter', String, callback)
    rospy.Subscriber('/cmd_vel', Twist, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
