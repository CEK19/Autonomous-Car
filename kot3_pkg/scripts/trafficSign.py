#! /usr/bin/python

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
import math
import numpy as np
import json
from keras import models
import pickle
import time

# index = 0
NODE_NAME_TRAFFIC_SIGN = rospy.get_param('NODE_NAME_TRAFFIC_SIGN')

TOPIC_NAME_CAMERA = rospy.get_param('TOPIC_NAME_CAMERA')

RESPONSE_SIGN = rospy.get_param('RESPONSE_SIGN')

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
	MIN_ACCURACY = 0.8
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


def threshold_RedSign(img, T_lower=125, T_upper=255):
	_, img = cv2.threshold(img, T_lower, T_upper, cv2.THRESH_BINARY)
	cv2.imshow("red threshhold", img)
	return img


def threshold_BlueSign(img, T_lower=100, T_upper=130):
	_, img = cv2.threshold(img, T_lower, T_upper, cv2.THRESH_BINARY)
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


def boundary_Box(img, contour, color=(0,255,0)):
	x, y, w, h = cv2.boundingRect(contour)
	extra = Sign.EXTRA_SAFETY
	img = cv2.rectangle(img, (x-extra, y-extra), (x+w+extra, y+h+extra), color, 5)
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
	prediction = model.predict(img)
	label = RESPONSE_SIGN['LABEL_TO_TEXT'][np.argmax(prediction)]
	accuracy = np.amax(prediction)
	return label, accuracy



def rosPublish(traffic_sign):
	rate = rospy.Rate(10) # 10hz
	# message = json.dumps({
	# 	"sign": MODULE_TRAFFIC_SIGNS.LABEL_TO_TEXT[traffic_sign],
	# 	"size": float(size),
	# 	"accuracy": float(accuracy)
	# })
	# print(message)
	# pub.publish(MODULE_TRAFFIC_SIGNS.LABEL_TO_TEXT[traffic_sign])
	pub.publish(traffic_sign)


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


def callbackFunction(data):
	print("---------------oke---------------")
	# global index
	bridge = CvBridge()
	print("Start Convert")
	imgMatrix = bridge.imgmsg_to_cv2(data, "bgr8")	#data.encoding

	# define the contrast and brightness value
	# contrast = 5. # Contrast control ( 0 to 127)
	# brightness = 2. # Brightness control (0-100)
	# imgMatrix = cv2.addWeighted( img, contrast, img, 0, brightness)

	# img = colorFilter(imgMatrix)
	# cv2.imshow("filtered", img)

	cv2.imshow("Origin", imgMatrix)
	cv2.waitKey(3)
	final_sign = []
	try:
		redness = returnRedness(imgMatrix)

		#### BLUE SIGN ####
		# print("start blue")
		thresh = threshold_BlueSign(redness)
		contours = findContour(thresh)
		for contour in contours:
			# print(contour)
			area = cv2.contourArea(contour)
			# print("area", area)
			x, y, w, h = cv2.boundingRect(contour)
			# print("x, y, w, h", x, y, w, h)
			
			# leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
			# rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
			# topmost = tuple(contour[contour[:, :, 1].argmin()][0])
			# bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
			# farest_h = distanceBetweenTwoPoint(topmost, bottommost)
			# farest_w = distanceBetweenTwoPoint(leftmost, rightmost)

			if (w <= Sign.MIN_WIDTH_HEIGHT or h <= Sign.MIN_WIDTH_HEIGHT):
				continue
			elif w/h >= Sign.WIDTH_HEIGHT_RATIO or h/w >= Sign.WIDTH_HEIGHT_RATIO:
				# print("blue not circle or square", w/h, h/w)
				# signTmp = boundary_Box(imgMatrix, contour)
				# cv2.imshow("temp", signTmp)
				# cv2.imshow("box", imgMatrix)
				continue
			# elif farest_w/farest_h >= 1.2 or farest_h/farest_w >= 1.2:
			# 	continue
			# elif farest_h/h >= 1.1 or h/farest_h >= 1.1:
			# 	continue
			# elif farest_w/w >= 1.1 or w/farest_w >= 1.1:
			# 	continue
			elif area > Sign.MIN_AREA and area < Sign.MAX_AREA:  # 15000
				print("from blue")
				final_sign.append(contour)
				# print("append blue success")
			else:
				# print('blue area: ', area)
				continue
		# print("end blue")

		#### RED SIGN ####
		# print("start red")
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
			elif w/h >= Sign.WIDTH_HEIGHT_RATIO or h/w >= Sign.WIDTH_HEIGHT_RATIO:
				# print("blue not circle or square", w/h, h/w)
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
				# print('red area: ', area)
				continue
		# print("end red")

		#### FINAL SIGN ####
		if final_sign:  # non-empty
			big = findBiggestContour(final_sign)
			sign = boundary_Box(imgMatrix, big)
			cv2.imshow('final', imgMatrix)
			startTime = time.time()
			prediction, accuracy = predict(sign)
			endTime = time.time()
			print("Time: ", endTime-startTime)
			print("Prediction: ", prediction, accuracy)
			size = cv2.contourArea(big)

			if (size >= Sign.MAX_AREA_TODETECT):
				if (len(potentialSign) > 1):
					rosPublish(potentialSign)
				potentialSign = []

			elif (accuracy > Sign.MIN_ACCURACY):
				potentialSign.append(prediction)
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
rospy.init_node(NODE_NAME_TRAFFIC_SIGN)
while not rospy.is_shutdown():
	lis = rospy.Subscriber(TOPIC_NAME_CAMERA, Image, callbackFunction)
	rospy.spin()