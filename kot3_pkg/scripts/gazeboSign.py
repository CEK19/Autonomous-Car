#! /usr/bin/python

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
import math
import numpy as np
import json
from os import listdir
from os.path import isfile, join
from numpy.core.fromnumeric import amax
from numpy.lib.type_check import imag
from keras import models
import pickle
import time

# index = 0


# CONST FILE

class Setting:
	MODEL_NAME = "model-110.h5"
	MODEL_PATH = "./assets/"
	DATA_PATH = "./assets/"
	
class Sign:
	EXTRA_SAFETY = 5
	MIN_AREA = 100
	MAX_AREA = 50000
	MIN_WIDTH_HEIGHT = 30
	MIN_ACCURACY = 0.2


class MODULE_TRAFFIC_SIGNS:
	AHEAD = "AHEAD"
	FORBID = "FORBID"
	STOP = "STOP"
	LEFT = "LEFT"
	RIGHT = "RIGHT"
	NONE = "NONE"
	LABEL_TO_TEXT = [AHEAD, FORBID, STOP, LEFT, RIGHT, NONE]

NODE_NAME_TRAFFIC_SIGNS = 'traffic_signs_node_name'
TOPIC_NAME_CAMERA = '/camera/rgb/image_raw'
	
#################################################

pub = rospy.Publisher('chatter', String, queue_size=1)

with open('/home/minhtu/NCKH_workspace/KOT3_ws/src/kot3_pkg/scripts/assets/mean_image_gray.pickle', 'rb') as f:
    MEAN_IMAGE = pickle.load(f, encoding='latin1')

with open('/home/minhtu/NCKH_workspace/KOT3_ws/src/kot3_pkg/scripts/assets/std_gray.pickle', 'rb') as f:
    STD_IMAGE = pickle.load(f, encoding='latin1')

#################################################


def distanceBetweenTwoPoint(PointA, PointB):
	return math.sqrt((PointA[0] - PointB[0])**2 + (PointA[1] - PointB[1])**2)


def returnRedness(img):
	yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(yuv)
	return v
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# blur = cv2.GaussianBlur(gray, (5, 5), 0)
	# cv2.imshow("blur", blur)
	# return blur

# T=145 - long range
# T=150 - short range
# T=160
# from T=128 to T=150 -> red sign detection -> best:150
# T= < T=120 -> blue sign detection -> best: 110


def threshold_RedSign(img, T=150):
	_, img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
	# cv2.imshow("red threshhold", img)
	return img

# T = 125	# data 2
# T = 110
# T = 100	# data 3
# T = 122
# T = 120
def threshold_BlueSign(img, T=110):
	_, img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
	# cv2.imshow("blue threshold", img)
	return img


def findContour(img):
	contours, hierarchy = cv2.findContours(
		img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours


def findBiggestContour(contours):
	m = 0
	c = [cv2.contourArea(i) for i in contours]
	return contours[c.index(max(c))]


def boundary_Green_Box(img, contours):
	x, y, w, h = cv2.boundingRect(contours)
	extra = Sign.EXTRA_SAFETY
	# img = cv2.rectangle(img, (x-5, y-5), (x+w+4, y+h+4), (0, 255, 0), 10)
	# sign = img[(y-5):(y+h+4), (x-5):(x+w+4)]
	img = cv2.rectangle(img, (x-extra, y-extra), (x+w+extra, y+h+extra), (0, 255, 0), 5)
	sign = img[(y-extra):(y+h+extra), (x-extra):(x+w+extra)]
	return sign


def preprocessingImageToClassifier(image=None, imageSize=32):
	# GRAYSCALE
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# RESIZE
	image = cv2.resize(image, (imageSize, imageSize))
	# LOCAL HISTOGRAM EQUALIZATION
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
	image = clahe.apply(image)
	image = image.astype(np.float32)/255.0
	cv2.imshow("classifyImg", image)
	image = image.reshape(1, imageSize, imageSize, 1)
	return image


def preprocessingImageToClassifierV2(image=None, imageSize=32):
	# RESIZE
	image = cv2.resize(image, (imageSize, imageSize))

	# GRAYSCALE
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# LOCAL HISTOGRAM EQUALIZATION
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
	image = clahe.apply(image)

	# /255.0 NORMALIZATION
	image = image.astype(np.float32)/255.0

	# Mean Normalization
	image = image - MEAN_IMAGE["mean_image_gray"]

	# STD Normalization
	image = image / STD_IMAGE["std_gray"]

	image = image.reshape(1, imageSize, imageSize, 1)
	return image


def predict(sign):
	img = preprocessingImageToClassifierV2(sign, imageSize=32)
	start_time = time.time()
	ans = model.predict(img)
	end_time = time.time()
	return ans, end_time-start_time



def rosPublish(traffic_sign, size, accuracy):
	rate = rospy.Rate(10) # 10hz
	message = json.dumps({
		"sign": MODULE_TRAFFIC_SIGNS.LABEL_TO_TEXT[traffic_sign],
		"size": float(size),
		"accuracy": float(accuracy)
	})
	print(message)
	pub.publish(message)


def callbackFunction(data):
	print("---------------oke---------------")
	# global index
	bridge = CvBridge()
	print("Start Convert")
	imgMatrix = bridge.imgmsg_to_cv2(data, "bgr8")	#data.encoding

	cv2.imshow("Origin", imgMatrix)
	cv2.waitKey(3)
	final_sign = []
	try:
		redness = returnRedness(imgMatrix)
		#### BLUE SIGN ####
		thresh = threshold_BlueSign(redness)
		contours = findContour(thresh)
		for contour in contours:
			area = cv2.contourArea(contour)
			x, y, w, h = cv2.boundingRect(contour)
			# isConvext = cv2.isContourConvex(contour)
			leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
			rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
			topmost = tuple(contour[contour[:, :, 1].argmin()][0])
			bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
			farest_h = distanceBetweenTwoPoint(topmost, bottommost)
			farest_w = distanceBetweenTwoPoint(leftmost, rightmost)
			if (w <= Sign.MIN_WIDTH_HEIGHT or h <= Sign.MIN_WIDTH_HEIGHT):
				continue
			# elif isConvext:
			# 	continue
			elif w/h >= 1.2 or h/w >= 1.2:
				continue
			elif farest_w/farest_h >= 1.2 or farest_h/farest_w >= 1.2:
				continue
			elif farest_h/h >= 1.1 or h/farest_h >= 1.1:
				continue
			elif farest_w/w >= 1.1 or w/farest_w >= 1.1:
				continue
			elif area > Sign.MIN_AREA and area < Sign.MAX_AREA:  # 15000
				print("from blue")
				final_sign.append(contour)
			else:
				continue

		#### RED SIGN ####
		thresh = threshold_RedSign(redness)
		contours = findContour(thresh)
		for contour in contours:
			area = cv2.contourArea(contour)
			x, y, w, h = cv2.boundingRect(contour)
			# isConvext = cv2.isContourConvex(contour)
			leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
			rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
			topmost = tuple(contour[contour[:, :, 1].argmin()][0])
			bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
			farest_h = distanceBetweenTwoPoint(topmost, bottommost)
			farest_w = distanceBetweenTwoPoint(leftmost, rightmost)
			if (w <= Sign.MIN_WIDTH_HEIGHT or h <= Sign.MIN_WIDTH_HEIGHT):
				continue
			# elif isConvext:
			# 	continue
			elif w/h >= 1.2 or h/w >= 1.2:
				continue
			elif farest_w/farest_h >= 1.2 or farest_h/farest_w >= 1.2:
				continue
			elif farest_h/h >= 1.1 or h/farest_h >= 1.1:
				continue
			elif farest_w/w >= 1.1 or w/farest_w >= 1.1:
				continue
			elif area > Sign.MIN_AREA and area < Sign.MAX_AREA:  # 15000
				print("from red")
				final_sign.append(contour)
			else:
				continue

		#### FINAL SIGN ####
		if final_sign:  # non-empty
			big = findBiggestContour(final_sign)
			sign = boundary_Green_Box(imgMatrix, big)
			cv2.imshow('final', imgMatrix)
			startTime = time.time()
			prediction, t = predict(sign)
			endTime = time.time()
			accuracy = np.amax(prediction)
			size = cv2.contourArea(big)
			rosPublish(np.argmax(prediction), size, accuracy)
		else:
			cv2.imshow('final', imgMatrix)
			print("I can't see anything !")

	except:
		print("err")

###########################
### MAIN FUNCTION	###
###########################


###########################


print("---------------begin---------------")
model = models.load_model('/home/minhtu/NCKH_workspace/KOT3_ws/src/kot3_pkg/scripts/assets/model-110.h5')
rospy.init_node(NODE_NAME_TRAFFIC_SIGNS)
while not rospy.is_shutdown():
	lis = rospy.Subscriber(TOPIC_NAME_CAMERA, Image, callbackFunction)
	rospy.spin()
