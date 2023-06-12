import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Twist
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import sys

print(sys.version)


class Nodo(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()

        # Publishers
        self.pub = rospy.Publisher('cmd_vel', Twist,queue_size=10)

        # Subscribers
        rospy.Subscriber("/camera/rgb/image_raw/compressed",Image,self.callback)


    def callback(self, msg):
        self.image = self.br.imgmsg_to_cv2(msg)
        twist = Twist()
        twist.linear.x = 0.5; twist.linear.y = 0.0; twist.linear.z = 0.0
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
        self.pub.publish(twist)


    def start(self):
        rospy.loginfo("Timing images")
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node("imagetimer111", anonymous=True)
    my_node = Nodo()
    my_node.start()