import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import socket
import cv2
import os
import numpy as np
import sys
import time
import json

print(sys.version)


class Nodo(object):
    def __init__(self):
        # Params
        self.preVal = 0.
        self.image = None
        self.br = CvBridge()

        # Publishers
        self.pub = rospy.Publisher('lane_detection_topic', String,queue_size=1)

        # Subscribers
        rospy.Subscriber("/camera/image",Image,self.callback)
        HOST = '192.168.1.4'  
        PORT = 8000        

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (HOST, PORT)
        print('connecting to %s port ' + str(server_address))
        self.s.connect(server_address)
        self.preTime = time.time()
        print("Init completed !")

        self.totalTime = 0
        self.counter = 1


    def callback(self, msg):
        timeDelta = time.time() - self.preTime
        
        self.preTime = time.time()
        if (timeDelta > 0.1) or (timeDelta < 0.02):
            return
        startTime = time.time()
        self.image = self.br.imgmsg_to_cv2(msg)
    
        frame = cv2.resize(self.image ,(128,128))
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

        msg = frame.tobytes()
        try:
            self.s.sendall(msg)
            data = self.s.recv(1024)
        except:
            print("TCP break, ROS will exit now")
            rospy.signal_shutdown("Exit by Error !")

        if data.decode("utf8") == "EXIT":
            print("request Exit by Lane server !")
            rospy.signal_shutdown("request Exit by Lane server !")
            return
        data = data.decode("utf8")
        print("recive ",data)
        if len(data) > 30:
            self.pub.publish(data)
        self.totalTime += time.time() - startTime
        print("total time: ",self.totalTime/self.counter)
        self.counter += 1

    def start(self):
        rospy.loginfo("Started")
        rospy.spin()
            

if __name__ == '__main__':
    rospy.init_node("imagetimer111", anonymous=True)
    my_node = Nodo()
    my_node.start()