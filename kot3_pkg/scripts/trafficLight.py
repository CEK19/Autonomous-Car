#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import json
import numpy as np
import cv2

# Test here
NODE_NAME_TRAFFIC_LIGHT = rospy.get_param('NODE_NAME_TRAFFIC_LIGHT')

TOPIC_NAME_CAMERA = rospy.get_param('TOPIC_NAME_CAMERA')
TOPIC_NAME_TRAFFIC_LIGHT = rospy.get_param('TOPIC_NAME_TRAFFIC_LIGHT')

pub = rospy.Publisher(TOPIC_NAME_TRAFFIC_LIGHT, String, queue_size=1)

#########
# Const #
#########
class Const:
	class TRAFFIC_LIGHT:
		red = "red"
		yellow = "yellow"
		green = "green"
		none = "none"

	class STANDARD_PROPERTY:
		minArea = 100
		maxArea = 30000
		widthHeightRatio = 1.8

	class PUBLISH_PROPERTY:
		minRange = 20000
		maxRange = 30000	# Equal to STANDARD_PROPERTY.maxArea

	COLOR_THRESHOLD = {
		"yellow": {  # done
			"sensitivity": np.array([]),
			"lower": np.array([15, 20, 230]),  # H,S,V
			"upper": np.array([30, 255, 255])
		},
		"red": {  # demo
			"sensitivity": np.array([170, 0, 0]),		# [170, 0, 0]
			# [0, 210, 110]	[0, 228, 124]
			"lower": np.array([0, 100, 110]),
			"upper": np.array([7, 150, 255])
		},
		"green": {  # done
			"sensitivity": np.array([]),
			"lower": np.array([48, 119, 125]),
			"upper": np.array([93, 255, 215])
		}
	}


class COLOR:
	green = (0, 255, 0)
	red = (0, 0, 255)
	yellow = (0, 255, 255)
	white = (255, 255, 255)


#########
# Utils #
#########
class Utils:
	@staticmethod
	def boundaryBox(img, obj, color):
		img = cv2.rectangle(img, (obj.x-5, obj.y-5),
							(obj.x+obj.w+4, obj.y+obj.h+4), color, 10)
		sign = img[(obj.y-5):(obj.y+obj.h+4), (obj.x-5):(obj.x+obj.w+4)]
		return sign

	@staticmethod
	def putText(img, text, color):
		cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=color)

	@staticmethod
	def publicVelocity(straight, angular):
		# myTwist = Twist()
		# myTwist.linear.x = straight
		# myTwist.angular.z = angular
		# pub.publish(myTwist)
		msg = json.dumps({"linear": straight, "angular": angular})
		pub.publish(msg)


class Light:
	def __init__(self, color):
		self.x = 0
		self.y = 0
		self.w = 0
		self.h = 0
		self.color = color
		self.contour = []
		self.size = 0
		return

	def setNewValue(self, contour, size, x, y, w, h):
		print(self.color, ": ", size)
		if size > self.size:
			self.contour = contour
			self.size = size
			self.x = x
			self.y = y
			self.w = w
			self.h = h
		return


class TrafficLight:
	def __init__(self):
		self.color = None
		self.red = Light(Const.TRAFFIC_LIGHT.red)
		self.yellow = Light(Const.TRAFFIC_LIGHT.yellow)
		self.green = Light(Const.TRAFFIC_LIGHT.green)

	def setNewValue(self, color, contour, areaSize, x, y, w, h):
		if color == Const.TRAFFIC_LIGHT.red:
			self.red.setNewValue(contour, areaSize, x, y, w, h)
		elif color == Const.TRAFFIC_LIGHT.green:
			self.green.setNewValue(contour, areaSize, x, y, w, h)
		elif color == Const.TRAFFIC_LIGHT.yellow:
			self.yellow.setNewValue(contour, areaSize, x, y, w, h)

	def checkLightProperty(self, img, color):
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours:
			area = cv2.contourArea(contour)
			x, y, w, h = cv2.boundingRect(contour)
			if area < Const.STANDARD_PROPERTY.minArea or area > Const.STANDARD_PROPERTY.maxArea:
				if area > Const.STANDARD_PROPERTY.maxArea:
					print("######################################", area)
				continue
			elif w/h >= Const.STANDARD_PROPERTY.widthHeightRatio or h/w >= Const.STANDARD_PROPERTY.widthHeightRatio:
				print("===================================>", w/h, h/w)
				continue
			# pass all standard property
			self.setNewValue(color, contour, area, x, y, w, h)

	def singleLightDetect(self, img: cv2.Mat, color: str):
		hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		sensitivity = Const.COLOR_THRESHOLD[color]["sensitivity"]
		lowerColor = Const.COLOR_THRESHOLD[color]["lower"]
		upperColor = Const.COLOR_THRESHOLD[color]["upper"]
		maskColor = []
		if len(sensitivity) == 0:
			maskColor = cv2.inRange(hsvImg, lowerColor, upperColor)
		else:
			maskColor0 = cv2.inRange(hsvImg, lowerColor, upperColor)
			maskColor1 = cv2.inRange(hsvImg, lowerColor + sensitivity, upperColor + sensitivity)
			maskColor = cv2.bitwise_or(maskColor0, maskColor1)

		cv2.imshow("mask-" + color, maskColor)
		self.checkLightProperty(maskColor, color)
		pass

	def classify(self, img):
		redSize = self.red.size
		greenSize = self.green.size
		yellowSize = self.yellow.size
		print("size: ", redSize, greenSize, yellowSize)
		if (redSize >= greenSize and redSize >= yellowSize and redSize != 0):
			print("red")
			self.color = Const.TRAFFIC_LIGHT.red
			sign = Utils.boundaryBox(img, self.red, COLOR.red)
			Utils.putText(img, "Traffic light detected: red", COLOR.red)
			cv2.imshow("final", img)
			cv2.imshow('sign', sign)
			return Const.TRAFFIC_LIGHT.red, redSize

		elif (greenSize >= redSize and greenSize >= yellowSize and greenSize != 0):
			print("green")
			self.color = Const.TRAFFIC_LIGHT.green
			sign = Utils.boundaryBox(img, self.green, COLOR.green)
			Utils.putText(img, "Traffic light detected: green", COLOR.green)
			cv2.imshow("final", img)
			cv2.imshow('sign', sign)
			return Const.TRAFFIC_LIGHT.green, greenSize

		elif (yellowSize >= redSize and yellowSize >= greenSize and yellowSize != 0):
			print("yellow")
			self.color = Const.TRAFFIC_LIGHT.yellow
			Utils.putText(img, "Traffic light detected: yellow", COLOR.yellow)
			sign = Utils.boundaryBox(img, self.yellow, COLOR.yellow)
			cv2.imshow("final", img)
			cv2.imshow('sign', sign)
			return Const.TRAFFIC_LIGHT.yellow, yellowSize

		else:
			self.color = None
			Utils.putText(img, "Traffic light detected: nothing", COLOR.white)
			cv2.imshow("final", img)
			return Const.TRAFFIC_LIGHT.none, 0


def trafficLightDetector(data):
	bridge = CvBridge()
	orgImg = bridge.imgmsg_to_cv2(data, "bgr8")
	cv2.imshow("org", orgImg)
	trafficLight = TrafficLight()
	trafficLight.singleLightDetect(orgImg, Const.TRAFFIC_LIGHT.green)
	trafficLight.singleLightDetect(orgImg, Const.TRAFFIC_LIGHT.red)
	trafficLight.singleLightDetect(orgImg, Const.TRAFFIC_LIGHT.yellow)
	# print(trafficLight.red.size)
	# print(trafficLight.green.size)
	# print(trafficLight.yellow.size)
	color, size = trafficLight.classify(orgImg)

	if color == Const.TRAFFIC_LIGHT.none:
		pub.publish(color)
	elif Const.PUBLISH_PROPERTY.minRange <= size <= Const.PUBLISH_PROPERTY.maxRange:
		pub.publish(color)
	cv2.waitKey(1)



if __name__ == '__main__':
	try:
		rospy.init_node(NODE_NAME_TRAFFIC_LIGHT, anonymous=True)
		rospy.Subscriber(TOPIC_NAME_CAMERA, Image, trafficLightDetector)

		rospy.spin()
	except rospy.ROSInterruptException:
		pass
