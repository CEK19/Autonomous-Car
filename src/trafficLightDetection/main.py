# 18/09/2022

import os
import cv2
import numpy as np
from const import *


def boundaryBox(img, obj, color):
	img = cv2.rectangle(img, (obj.x-5, obj.y-5), (obj.x+obj.w+4, obj.y+obj.h+4), color, 10)
	sign = img[(obj.y-5):(obj.y+obj.h+4), (obj.x-5):(obj.x+obj.w+4)]
	return sign


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
		self.red = Light(TRAFFIC_LIGHT.red)
		self.yellow = Light(TRAFFIC_LIGHT.yellow)
		self.green = Light(TRAFFIC_LIGHT.green)

	def setNewValue(self, color, contour, areaSize, x, y, w, h):
		if color == TRAFFIC_LIGHT.red:
			self.red.setNewValue(contour, areaSize, x, y, w, h)
		elif color == TRAFFIC_LIGHT.green:
			self.green.setNewValue(contour, areaSize, x, y, w, h)
		elif color == TRAFFIC_LIGHT.yellow:
			self.yellow.setNewValue(contour, areaSize, x, y, w, h)

	def checkLightProperty(self, img, color):
		contours, hierarchy = cv2.findContours(
			img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours:
			area = cv2.contourArea(contour)
			x, y, w, h = cv2.boundingRect(contour)
			if area < Setting.STANDARD_PROPERTY.minArea or area > Setting.STANDARD_PROPERTY.maxArea:
				continue
			elif w/h >= Setting.STANDARD_PROPERTY.widthHeightRatio or h/w >= Setting.STANDARD_PROPERTY.widthHeightRatio:
				continue
			# pass all standard property
			self.setNewValue(color, contour, area, x, y, w, h)

	def singleLightDetect(self, img: cv2.Mat, color: str):
		hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		sensitivity = Setting.COLOR_THRESHOLD[color]["sensitivity"]
		lowerColor = Setting.COLOR_THRESHOLD[color]["lower"]
		upperColor = Setting.COLOR_THRESHOLD[color]["upper"]
		maskColor = []
		if len(sensitivity) == 0:
			maskColor = cv2.inRange(hsvImg, lowerColor, upperColor)
		else:
			maskColor0 = cv2.inRange(hsvImg, lowerColor, upperColor)
			maskColor1 = cv2.inRange(
				hsvImg, lowerColor + sensitivity, upperColor + sensitivity)
			maskColor = cv2.bitwise_or(maskColor0, maskColor1)
		# print(maskColor)
		cv2.imshow("mask-" + color, maskColor)
		self.checkLightProperty(maskColor, color)
		pass

	def classify(self, img):
		redSize = self.red.size
		greenSize = self.green.size
		yellowSize = self.yellow.size
		print("size: ", redSize, greenSize, yellowSize)
		if (redSize >= greenSize and redSize >= yellowSize and redSize != 0):
			print("red");
			self.color = TRAFFIC_LIGHT.red
			sign = boundaryBox(img, self.red, COLOR.red)
			cv2.putText(img, "Traffic light detected: red", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=COLOR.red)
			cv2.imshow("final", img)
			cv2.imshow('sign', sign)
		elif (greenSize >= redSize and greenSize >= yellowSize and greenSize != 0):
			print("green")
			self.color = TRAFFIC_LIGHT.green
			sign = boundaryBox(img, self.green, COLOR.green)
			cv2.putText(img, "Traffic light detected: green", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=COLOR.green)
			cv2.imshow("final", img)
			cv2.imshow('sign', sign)
		elif (yellowSize >= redSize and yellowSize >= greenSize and yellowSize != 0):
			print("yellow")
			self.color = TRAFFIC_LIGHT.yellow
			cv2.putText(img, "Traffic light detected: yellow", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=COLOR.yellow)
			sign = boundaryBox(img, self.yellow, COLOR.yellow)
			cv2.imshow("final", img)
			cv2.imshow('sign', sign)
		else:
			self.color = None
			cv2.putText(img, "Traffic light detected: nothing", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=COLOR.white)
			cv2.imshow("final", img)


def readImg(fileName: str):
	return cv2.imread(fileName)


def binaryImg1(img: cv2.Mat, T: int = 30):
	grayImg: cv2.Mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurredImg: cv2.Mat = cv2.GaussianBlur(grayImg, (7, 7), 0)
	blurredImg = grayImg
	# cv2.imshow("blurredImg", blurredImg)
	_, threshImg = cv2.threshold(blurredImg, T, 255, cv2.THRESH_BINARY_INV)
	return threshImg


def binaryImg2(img: cv2.Mat):
	grayImg: cv2.Mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurredImg: cv2.Mat = cv2.GaussianBlur(grayImg, (3, 3), 0)
	threshImg = cv2.adaptiveThreshold(
		blurredImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
	return threshImg


def binaryImg3(img: cv2.Mat):
	grayImg: cv2.Mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurredImg: cv2.Mat = cv2.GaussianBlur(grayImg, (3, 3), 0)
	# blurredImg = grayImg
	threshImg = cv2.adaptiveThreshold(
		blurredImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
	return threshImg


def binaryImg4(img: cv2.Mat):
	grayImg: cv2.Mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurredImg: cv2.Mat = cv2.GaussianBlur(grayImg, (3, 3), 0)
	# blurredImg = grayImg
	threshImg = cv2.Canny(blurredImg, 200, 255)
	return threshImg


def canyEdge(img: cv2.Mat):
	copy = img.copy()
	thresh = binaryImg2(img)
	cannyEdgeDectection = cv2.Canny(thresh, 200, 255)
	cv2.imshow("cany", cannyEdgeDectection)
	imagecontours, _ = cv2.findContours(
		thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in imagecontours:
		if (cv2.contourArea(cnt) > 100):
			cv2.drawContours(copy, cnt, -1, (0, 255, 255), 2)
	cv2.imshow("thre", thresh)
	cv2.imshow("cnt", copy)



############
#   MAIN   #
############
# orgImg = readImg("C:/Users/Admin/Documents/Tu/coding/Autonomous-Car/src/trafficLightDetection/assets/green1.jpg")
# orgImg = cv2.imread("C:\\Users\\Admin\\Documents\\coding\\Autonomous-Car\\src\\trafficLightDetection\\assets\\red1.jpg")

# print()
# print(Setting.COLOR_THRESHOLD[0])
# print()

if Setting.MODE == Mode.PIC:
	orgImg = cv2.imread(Setting.PICTURE_PATH)
	cv2.imshow("org", orgImg)
	trafficLight = TrafficLight()
	trafficLight.singleLightDetect(orgImg, "green")
	trafficLight.singleLightDetect(orgImg, "red")
	trafficLight.singleLightDetect(orgImg, "yellow")
	print(trafficLight.red.size)
	print(trafficLight.green.size)
	print(trafficLight.yellow.size)
	trafficLight.classify(orgImg)
	cv2.waitKey(0)
 
elif Setting.MODE == Mode.CAMERA:
	cam = cv2.VideoCapture(0)
	while True:
		ret, frame = cam.read()
		key = cv2.waitKey(1)
		if key == ord('q'):
			break

		trafficLight = TrafficLight()
		trafficLight.singleLightDetect(frame, "green")
		trafficLight.singleLightDetect(frame, "red")
		trafficLight.singleLightDetect(frame, "yellow")
		print(trafficLight.red.size)
		print(trafficLight.green.size)
		print(trafficLight.yellow.size)
		trafficLight.classify(frame)
	cv2.destroyAllWindows()
		
elif Setting.MODE == Mode.VIDEO:
	status = ''
	while True:
		vid = cv2.VideoCapture(Setting.PATH)
		currentframe = 0

		while True:
			ret, frame = vid.read()

			key = cv2.waitKey(1)
			if key == ord('q'):
				status = 'quit'
				break
			elif key == ord('p'):
				cv2.waitKey(-1)
			elif key == ord('r'):
				status = 'replay'
				break

			if ret:
				trafficLight = TrafficLight()
				trafficLight.singleLightDetect(frame, "green")
				trafficLight.singleLightDetect(frame, "red")
				trafficLight.singleLightDetect(frame, "yellow")
				print(trafficLight.red.size)
				print(trafficLight.green.size)
				print(trafficLight.yellow.size)
				trafficLight.classify(frame)
				
				currentframe += 1
				print(currentframe)
			else:
				print("end")
				status = 'end'
				break
		print("vid release")
		vid.release()
		if status == 'quit' or status == 'end':
			print("break")
			break
	print("out")
	cv2.destroyAllWindows()

