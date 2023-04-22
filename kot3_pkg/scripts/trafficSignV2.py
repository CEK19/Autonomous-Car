#! /usr/bin/python

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
import math
import numpy as np
import pickle
import time
import json
from keras import models
from ultralytics import YOLO

# index = 0
NODE_NAME_TRAFFIC_SIGN = rospy.get_param('NODE_NAME_TRAFFIC_SIGN')

TOPIC_NAME_CAMERA = rospy.get_param('TOPIC_NAME_CAMERA')

RESPONSE_SIGN = rospy.get_param('RESPONSE_SIGN')


# MODULE FILE
CNNmodel = models.load_model('/home/minhtu/NCKH_workspace/KOT3_ws/src/kot3_pkg/scripts/assets/model-110.h5')
YOLOmodel = YOLO('/home/minhtu/NCKH_workspace/KOT3_ws/src/kot3_pkg/scripts/assets/yolo-model.pt')

# CONST FILE
class Setting:
	MODEL_NAME = "model-110.h5"
	MODEL_PATH = "./assets/"
	DATA_PATH = "./assets/"
	
class Sign:
	EXTRA_SAFETY = 5
	MIN_AREA = 100
	MAX_AREA = 50000
	MAX_AREA_TODETECT = 40000
	MIN_WIDTH_HEIGHT = 30
	MIN_ACCURACY = 0.75
	WIDTH_HEIGHT_RATIO = 1.3
 

### change to read in yaml file
# class MODULE_TRAFFIC_SIGNS:
#     	AHEAD = "AHEAD"
# 	FORBID = "FORBID"
# 	STOP = "STOP"
# 	LEFT = "LEFT"
# 	RIGHT = "RIGHT"
# 	NONE = "NONE"
# 	LABEL_TO_TEXT = [AHEAD, FORBID, STOP, LEFT, RIGHT, NONE]

class ColorThreshold:
    class RED:
        sensitivity = np.array([170, 0, 0])
        lower = np.array([0, 95, 110])     # [0, 113, 150]
        upper = np.array([10, 255, 255])    # [10, 255, 255]
    class BLUE:
        sensitivity = np.array([0, 0, 0])
        lower = np.array([95, 100, 100])      # [90, 50, 70]
        upper = np.array([128, 255, 255])   # [128, 255, 255]

	
#################################################
# TODO later, an dia r ghep
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
	cv2.imshow("red threshhold", img)
	return img

# T = 125	# data 2
# T = 110
# T = 100	# data 3
# T = 122
# T = 120
def threshold_BlueSign(img, T=110):
	_, img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
	cv2.imshow("blue threshold", img)
	return img


def findContour(img):
	contours, hierarchy = cv2.findContours(
		img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours


def findBiggestContour(contours):
	m = 0
	c = [cv2.contourArea(i) for i in contours]
	return contours[c.index(max(c))]


def boundary_Green_Box(img, contour):
	x, y, w, h = cv2.boundingRect(contour)
	extra = Sign.EXTRA_SAFETY
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
	ans = CNNmodel.predict(img)
	end_time = time.time()
	return ans, end_time-start_time



def rosPublish(traffic_sign):
	rate = rospy.Rate(10) # 10hz
	# message = json.dumps({
	# 	"sign": MODULE_TRAFFIC_SIGNS.LABEL_TO_TEXT[traffic_sign],
	# 	"size": float(size),
	# 	"accuracy": float(accuracy)
	# })
	# print(message)
	# pub.publish(MODULE_TRAFFIC_SIGNS.LABEL_TO_TEXT[traffic_sign])
	pub.publish(RESPONSE_SIGN['LABEL_TO_TEXT'][traffic_sign])


def colorFilter(img):
	hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	sensitivity = ColorThreshold.RED.sensitivity
	
	# Keep red color
	maskColorRed1 = cv2.inRange(hsvImg, ColorThreshold.RED.lower, ColorThreshold.RED.upper)
	maskColorRed2 = cv2.inRange(hsvImg, ColorThreshold.RED.lower + sensitivity, ColorThreshold.RED.upper + sensitivity)
	maskColorRed = cv2.bitwise_or(maskColorRed1, maskColorRed2)
	cv2.imshow("maskColorRed", maskColorRed)
	
	# Keep blue color
	maskColorBlue = cv2.inRange(hsvImg, ColorThreshold.BLUE.lower, ColorThreshold.BLUE.upper)
	cv2.imshow("maskColorBlue", maskColorBlue)
	
	# Combine mask
	mask = cv2.bitwise_or(maskColorRed, maskColorBlue)
	cv2.imshow("mask", mask)
	
	# connect gaps in binary image (another branch, not used yet)
	contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		cv2.drawContours(mask, [cnt], 0, 255, 15)
		cv2.drawContours(mask, [cnt], 0, 255, -1)
	cv2.imshow("mask after contour", mask)
	
	
	resultColor = cv2.bitwise_and(img, img, mask=mask)
	cv2.imshow("resultColor", resultColor)
	cv2.imshow("orgImg", img)
	return resultColor


potentialSign = []
lastTime = time.time()


def callbackFunction(data):
	QTM_time_start = time.time()
	global lastTime
	global potentialSign

	print(lastTime, QTM_time_start, QTM_time_start - lastTime)
	if QTM_time_start - lastTime >= 0.1 or QTM_time_start - lastTime < 0.02:
		lastTime = QTM_time_start
		return
	lastTime = QTM_time_start
 
	print("---------------read img---------------")
	bridge = CvBridge()
	imgMatrix = bridge.imgmsg_to_cv2(data, "bgr8")	#data.encoding

	cv2.imshow("Origin", imgMatrix)
	cv2.waitKey(3)
	final_sign = []
	print("done read img")


	print("---------------detection---------------")
	# YOLO only detect traffic sign
	result = YOLOmodel(imgMatrix)[0]
	visual = result.plot()
	signs = result.boxes
	cv2.imshow("YOLO", visual)
	bigSize = 0
	x, y, w, h = 0, 0, 0, 0

	if not len(signs.conf):
		print("time: ", time.time()-QTM_time_start)
		return

	for i in range(len(signs.conf)):
		# print()
		print("My boxes:", signs)
		print()
		if signs.conf[i] < Sign.MIN_ACCURACY:
			print("low accuracy")
			print("time: ", time.time()-QTM_time_start)
			return

		# center point (x,y), width (w), height (h)
		xywh = (np.rint(signs.xywh[i].numpy())).astype(int)
		# Top left corner (x1,y1), bottom right corner (x2,y2)
		xyxy = (np.rint(signs.xyxy[i].numpy())).astype(int)
		width, height = xywh[2], xywh[3]
		if (width*height > bigSize):
			bigSize = width*height
			x = round( xyxy[0] )
			y = round( xyxy[1] )
			w = round( xywh[2] )
			h = round( xywh[3] )
			# big = signs

	print("---------------crop---------------")
	print(x, y, w, h)
	extra = 0	# Sign.EXTRA_SAFETY
	sign = imgMatrix[(y-extra):(y+h+extra), (x-extra):(x+w+extra)]


	# big = findBiggestContour(final_sign)
	# sign = boundary_Green_Box(imgMatrix, big)
	# cv2.imshow('final', imgMatrix)

	print("---------------classification---------------")
	cv2.imshow("sign", sign)
	startTime = time.time()
	prediction, t = predict(sign)
	endTime = time.time()
	print("total time: ", endTime-startTime, ", predict time:", t)
	label = RESPONSE_SIGN['LABEL_TO_TEXT'][np.argmax(prediction)]
	accuracy = np.amax(prediction)
	
	print("=====> label: ", label, ", accuracy: ", accuracy)

	if (bigSize >= Sign.MAX_AREA_TODETECT):
		if (len(potentialSign) > 1):
			rosPublish(label)
		potentialSign = []

	elif (accuracy > Sign.MIN_ACCURACY):
		potentialSign.append(label)
	
	print("time: ", time.time()-QTM_time_start)

###########################
### MAIN FUNCTION	###
###########################


###########################

print("---------------begin---------------")
rospy.init_node(NODE_NAME_TRAFFIC_SIGN)
while not rospy.is_shutdown():
	lis = rospy.Subscriber(TOPIC_NAME_CAMERA, Image, callbackFunction)
	rospy.spin()