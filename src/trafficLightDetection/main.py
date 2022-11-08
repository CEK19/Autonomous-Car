# 18/09/2022

import cv2
import numpy as np


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
	threshImg = cv2.adaptiveThreshold(blurredImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
	return threshImg


def binaryImg3(img: cv2.Mat):
	grayImg: cv2.Mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurredImg: cv2.Mat = cv2.GaussianBlur(grayImg, (3, 3), 0)
	# blurredImg = grayImg
	threshImg = cv2.adaptiveThreshold(blurredImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
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
	imagecontours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in imagecontours:
		if(cv2.contourArea(cnt) > 100):
			cv2.drawContours(copy, cnt, -1, (0, 255, 255), 2)
	cv2.imshow("thre", thresh)
	cv2.imshow("cnt", copy)

def colorFilter(img: cv2.Mat):
	hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	sensitivity = np.array([170, 0, 0])

	# red detection
	lowerRed = np.array([0, 113, 150])
	upperRed = np.array([10, 255, 255])
	maskRed1 = cv2.inRange(hsvImg, lowerRed, upperRed)
	maskRed2 = cv2.inRange(hsvImg, lowerRed + sensitivity, upperRed + sensitivity)
	maskRed = cv2.bitwise_or(maskRed1, maskRed2)
	# copyOriginalFrame = img.copy()
	# copyOriginalFrame[np.where(maskRed == [0])] = [0]
	# cv2.imshow("maskRed", copyOriginalFrame)

	# yellow detection
	lowerYellow = np.array([10, 135, 155])  # H,S,V
	upperYellow = np.array([25, 255, 255])  # H,S,V
	maskYellow = cv2.inRange(hsvImg, lowerYellow, upperYellow)
	# copyOriginalFrame = img.copy()
	# copyOriginalFrame[np.where(maskYellow == [0])] = [255]
	# cv2.imshow("maskYellow", copyOriginalFrame)

	# green detection
	lowerGreen = np.array([80, 50, 115])  # H,S,V
	upperGreen = np.array([90, 255, 255])  # H,S,V
	maskGreen = cv2.inRange(hsvImg, lowerGreen, upperGreen)
	# copyOriginalFrame = img.copy()
	# copyOriginalFrame[np.where(maskGreen == [0])] = [0]
	# cv2.imshow("maskGreen", copyOriginalFrame)

	# black detection
	lowerBlack = np.array([96, 0, 0])  # H,S,V
	upperBlack = np.array([180, 65, 110])  # H,S,V
	maskBlack = cv2.inRange(hsvImg, lowerBlack, upperBlack)
	# copyOriginalFrame = img.copy()
	# copyOriginalFrame[np.where(maskBlack == [0])] = [255]
	# cv2.imshow("maskBlack", copyOriginalFrame)

	yellowAndRed = cv2.bitwise_or(maskYellow, maskRed)
	maskAllColor = cv2.bitwise_or(yellowAndRed, maskGreen)

	allColor = img.copy()
	allColor[np.where(maskAllColor == [0])] = [0]
	cv2.imshow("all color", allColor)

	maskAll = allColor.copy()
	maskAll[np.where(maskBlack != [0])] = [255]
	cv2.imshow("mask all", maskAll)

	# cv2.imshow("result", copyOriginalFrame)

	# blurred = cv2.GaussianBlur(img, (3, 3), 0)
	# cannyEdgeDectection = cv2.Canny(blurred, 200, 255)
	# cv2.imshow('cany', cannyEdgeDectection)
	# combineFirst = cv2.bitwise_or(cannyEdgeDectection, maskAll)
	# cv2.imshow('gray', combineFirst)
	# backtorgb = cv2.cvtColor(combineFirst, cv2.COLOR_GRAY2RGB)
	cv2.imshow('return', cv2.vconcat([allColor, img]))
	# return cv2.vconcat([backtorgb, img])
	return allColor, maskAll


############
#   MAIN   #
############
# orgImg = readImg("C:/Users/Admin/Documents/Tu/coding/Autonomous-Car/src/trafficLightDetection/assets/green1.jpg")
orgImg = cv2.imread("C:\\Users\\Admin\\Documents\\coding\\Autonomous-Car\\src\\trafficLightDetection\\assets\\green2.jpg")


# b1 = binaryImg1(orgImg)
# cv2.imshow("b1", b1)

# b3 = binaryImg1(orgImg)
# cv2.imshow("b3", b3)

# b4 = binaryImg1(orgImg)
# cv2.imshow("b4", b4)

print(orgImg)

canyEdge(orgImg)

filteredImg, filteredImgV2 = colorFilter(orgImg)
cv2.imshow("filteredImg", filteredImgV2)
thresholdImg = binaryImg1(filteredImgV2, 2)
cv2.imshow("thresholdImg", thresholdImg)

imagecontours, _ = cv2.findContours(thresholdImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for count in imagecontours:
	area = cv2.contourArea(count)
	x, y, w, h = cv2.boundingRect(count)
	if(100 > area or area > 30000):
		continue
	print(cv2.contourArea(count))
	epsilon = 0.01 * cv2.arcLength(count, True)
	approximations = cv2.approxPolyDP(count, epsilon, True)
	i, j = approximations[0][0]
	print(len(approximations))
	condition1 = 6 < len(approximations) < 13
	condition2 = w/h <2  and h/w < 2
	if condition1 and condition2:
		cv2.putText(orgImg, "GREEN", (i-60, j-8), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
		cv2.drawContours(orgImg, [approximations], 0, (0, 0, 255), 3)
	# if len(approximations) >= 13:
	# 	 cv2.putText(orgImg, "other", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
	# 	 cv2.drawContours(orgImg, [approximations], 0, (0, 0, 255), 3)

cv2.imshow("final", orgImg)

cv2.waitKey(0)
