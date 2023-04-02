#!/usr/bin/env python

import rospy
from std_msgs.msg import String

NODE_NAME_INIT = rospy.get_param("NODE_NAME_INIT")

def callback(data):
    print("aaaa", data)
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)

def listener():

    rospy.init_node(NODE_NAME_INIT, anonymous=True)

    # rospy.Subscriber('chatter', String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
