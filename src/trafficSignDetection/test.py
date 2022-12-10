import cv2
import math
import numpy as np
from numpy.core.fromnumeric import amax
from numpy.lib.type_check import imag
from keras import models
import time
from const import *
from os import listdir
from os.path import isfile, join

# index = 0


def distanceBetweenTwoPoint(PointA, PointB):
	return math.sqrt((PointA[0] - PointB[0])**2 + (PointA[1] - PointB[1])**2)


def returnRedness(img):
	yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(yuv)
	return v

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
def threshold_BlueSign(img, T=130):
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


def boundary_Green_Box(img, contours):
	x, y, w, h = cv2.boundingRect(contours)
	extra = Sign.EXTRA_SAFETY
	# img = cv2.rectangle(img, (x-5, y-5), (x+w+4, y+h+4), (0, 255, 0), 10)
	# sign = img[(y-5):(y+h+4), (x-5):(x+w+4)]
	img = cv2.rectangle(img, (x-extra, y-extra), (x+w+extra, y+h+extra), (0, 255, 0), 5)
	sign = img[(y-extra):(y+h+extra), (x-extra):(x+w+extra)]
	return img, sign


def preprocessingImageToClassifier(image=None, imageSize=32):
	# GRAYSCALE
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# RESIZE
	image = cv2.resize(image, (imageSize, imageSize))
	cv2.imshow("resize", image)
	# LOCAL HISTOGRAM EQUALIZATION
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
	image = clahe.apply(image)
	image = image.astype(np.float32)/255.0
	image = image.reshape(1, imageSize, imageSize, 1)
	return image


def predict(sign):
	img = preprocessingImageToClassifier(sign, imageSize=32)
	start_time = time.time()
	ans = model.predict(img)
	end_time = time.time()
	return ans, end_time-start_time


def probility(sign):
	img = preprocessingImageToClassifier(sign, imageSize=32)
	return np.amax(model.predict(img))


labelToText = {
	0: "ahead",
	1: "forbid",
	2: "stop",
	3: "left",
	4: "right",
	5: "5",
	6: "6"
}

# labelToText = {
# 	0:"stop",
# 	1:"left",
# 	2:"right",
# }

# def publish(traffic_sign):
# 	pub = rospy.Publisher('chatter', custom_msg, queue_size=10)

# 	customMsg = custom_msg()
# 	customMsg.name = 1
# 	customMsg.num1 = traffic_sign
# 	customMsg.num2 = -1

# 	pub.publish(customMsg)


def callbackFunction(fileName, expectedResult=None, index=None):
	print("---------------oke---------------")
	# global index
	# bridge = CvBridge()
	print("Start Convert")
	# imgMatrix = bridge.imgmsg_to_cv2(data, "bgr8")	#data.encoding

	# basePath = "/home/minhtulehoang/catkin_ws/src/rotate_turtlebot/src/img"
	# baseFileName = "/pic" + str(index) + ".jpeg"
	imgMatrix = cv2.imread(fileName)
	cv2.imshow("Origin", imgMatrix)

	# print("Start Create File")
	#cv2.imwrite(basePath + baseFileName, imgMatrix)
	# print("Successful!")
	# index +=1
	# cv2.waitKey(3)
	final_sign = []
	redness = returnRedness(imgMatrix)
	try:
		#### RED SIGN ####
		thresh = threshold_RedSign(redness)
		contours = findContour(thresh)
		for contour in contours:
			area = cv2.contourArea(contour)
			x, y, w, h = cv2.boundingRect(contour)
			leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
			rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
			topmost = tuple(contour[contour[:, :, 1].argmin()][0])
			bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
			farest_h = distanceBetweenTwoPoint(topmost, bottommost)
			farest_w = distanceBetweenTwoPoint(leftmost, rightmost)
			if (w <= 20 or h <= 20):
				continue
			elif w/h >= 1.2 or h/w >= 1.2:
				continue
			elif farest_w/farest_h >= 1.2 or farest_h/farest_w >= 1.2:
				continue
			elif farest_h/h >= 1.1 or h/farest_h >= 1.1:
				continue
			elif farest_w/w >= 1.1 or w/farest_w >= 1.1:
				continue
			elif area > Sign.MIN_AREA and area < Sign.MAX_AREA:  # 15000
				area = cv2.contourArea(contour)
				print("from red")
				print(w/h)
				print(h/w)
				final_sign.append(contour)
			else:
				area = cv2.contourArea(contour)
				x, y, w, h = cv2.boundingRect(contour)
				print("NEARLY OKE, w=", w, " - h=", h, " - area= ", area)
				continue

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
			if (w <= 20 or h <= 20):
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

		#### FINAL SIGN ####
		if final_sign:  # non-empty
			big = findBiggestContour(final_sign)
			img, sign = boundary_Green_Box(imgMatrix, big)
			cv2.imshow('final', img)
			startTime = time.time()
			prediction, t = predict(sign)
			endTime = time.time()
   
			accuracy = np.amax(prediction)
			text = "Now,I see: " + labelToText[np.argmax(prediction)] + " with " + str(accuracy) + "%"
			timeText = "Total time: " + str(endTime - startTime) + ", predict time: " + str(t)
			cv2.putText(img, text, org=(10,25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 0))
			cv2.putText(img, timeText, org=(10,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 0))
			cv2.imshow('final', img)
			print("Now,I see:" + labelToText[np.argmax(prediction)] + " in ", endTime - startTime, "s")
			print(accuracy)

			x, y, w, h = cv2.boundingRect(big)
			print("width= ", w, " - height= ", h)
			print(cv2.isContourConvex(big))
			leftmost = tuple(big[big[:, :, 0].argmin()][0])
			rightmost = tuple(big[big[:, :, 0].argmax()][0])
			topmost = tuple(big[big[:, :, 1].argmin()][0])
			bottommost = tuple(big[big[:, :, 1].argmax()][0])
			print("farest_w= ", distanceBetweenTwoPoint(leftmost, rightmost))
			print("farest_h= ", distanceBetweenTwoPoint(topmost, bottommost))

			if ENABLE_WRITE_FILE and expectedResult != None:
				file = open(
					"C:\\Users\\Admin\\Documents\\coding\\masterAI\\traffic sign detection\\logs\\validate.txt", "a")
				if labelToText[np.argmax(prediction)] == expectedResult:
					file.write(str(index) + ": " + expectedResult +
							   " - accuracy: " + str(np.amax(prediction)) + "\n")
				else:
					file.write(str(index) + ": " + expectedResult + " --> " + labelToText[np.argmax(
						prediction)] + " - accuracy: " + str(np.amax(prediction)) + "\n")
				file.close()

			# publish(np.argmax(prediction))
		else:
			cv2.imshow('final', imgMatrix)
			print("I can't see anything !")
			if ENABLE_WRITE_FILE:
				file = open(
					"C:\\Users\\Admin\\Documents\\coding\\masterAI\\traffic sign detection\\logs\\validate.txt", "a")
				if expectedResult != None:
					file.write(str(index) + ": " + "see nothing\n")
				file.close()

	except:
		print("err")


def callbackFunctionVid(imgMatrix):
	final_sign = []
	redness = returnRedness(imgMatrix)
	try:
		#### RED SIGN ####
		thresh = threshold_RedSign(redness)
		contours = findContour(thresh)
		for contour in contours:
			area = cv2.contourArea(contour)
			x, y, w, h = cv2.boundingRect(contour)
			leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
			rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
			topmost = tuple(contour[contour[:, :, 1].argmin()][0])
			bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
			farest_h = distanceBetweenTwoPoint(topmost, bottommost)
			farest_w = distanceBetweenTwoPoint(leftmost, rightmost)
			if (w <= 20 or h <= 20):
				continue
			elif w/h >= 1.2 or h/w >= 1.2:
				continue
			elif farest_w/farest_h >= 1.2 or farest_h/farest_w >= 1.2:
				continue
			elif farest_h/h >= 1.1 or h/farest_h >= 1.1:
				continue
			elif farest_w/w >= 1.1 or w/farest_w >= 1.1:
				continue
			elif area > Sign.MIN_AREA and area < Sign.MAX_AREA:  # 15000
				final_sign.append(contour)
			else:
				continue

		#### BLUE SIGN ####
		thresh = threshold_BlueSign(redness, 120)
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
			if (w <= 20 or h <= 20):
				continue
			elif w/h >= 1.2 or h/w >= 1.2:
				continue
			elif farest_w/farest_h >= 1.2 or farest_h/farest_w >= 1.2:
				continue
			elif farest_h/h >= 1.1 or h/farest_h >= 1.1:
				continue
			elif farest_w/w >= 1.1 or w/farest_w >= 1.1:
				continue
			elif area > Sign.MIN_AREA and area < Sign.MAX_AREA:  # 15000
				final_sign.append(contour)
			else:
				continue

		#### FINAL SIGN ####
		if final_sign:  # non-empty
			big = findBiggestContour(final_sign)
			img, sign = boundary_Green_Box(imgMatrix, big)
			print("3")
			startTime = time.time()
			prediction, t = predict(sign)
			endTime = time.time()

			accuracy = np.amax(prediction)
			if (accuracy < 0.3):
				cv2.putText(img, "Now,I see: nothing !!", org=(10,25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=(0, 255, 0))
				cv2.imshow('final', imgMatrix)
				return
			print("4")
			text = "Now,I see: " + labelToText[np.argmax(prediction)] + " with " + str(accuracy) + "%"
			timeText = "Total time: " + str(endTime - startTime) + ", predict time: " + str(t)
			print(text)
			cv2.putText(img, text, org=(10,25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 0))
			cv2.putText(img, timeText, org=(10,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 0))
			cv2.imshow('final', img)
			# x, y, w, h = cv2.boundingRect(big)
			# print("width= ", w, " - height= ", h)
			# print(cv2.isContourConvex(big))
			# leftmost = tuple(big[big[:, :, 0].argmin()][0])
			# rightmost = tuple(big[big[:, :, 0].argmax()][0])
			# topmost = tuple(big[big[:, :, 1].argmin()][0])
			# bottommost = tuple(big[big[:, :, 1].argmax()][0])
			# print("farest_w= ", distanceBetweenTwoPoint(leftmost, rightmost))
			# print("farest_h= ", distanceBetweenTwoPoint(topmost, bottommost))

		else:
			cv2.putText(img, "nothing !!", org=(0,25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, color=(0, 255, 0))
			cv2.imshow('final', imgMatrix)

	except:
		print("err")


###########################
### MAIN FUNCTION	###
###########################

ENABLE_WRITE_FILE = True
MODE = Mode.CAMERA
# modelPath = "C:\\Users\\Admin\Documents\\coding\\masterAI\\traffic sign detection\\models"
modelPath = "./models"
videoPath = "/Users/lap15864-local/Desktop/tempVid.mov"

###########################

if MODE == Mode.ROS:
	model = models.load_model(modelPath + "\\" + "tkll-model.h5")
	pass
	# while not rospy.is_shutdown():
	# 	print("---------------begin---------------")
	# 	rospy.init_node("topic_receiver_img")
	# 	lis = rospy.Subscriber("/camera/rgb/image_raw", Image, callbackFunction)
	# 	rospy.spin()

if MODE == Mode.PICS_IN_FOLDER:
	onlymodels = [f for f in listdir(modelPath) if isfile(join(modelPath, f))]
	for index, modelName in enumerate(onlymodels):
		# if index == 2:
		# 	break
		# if modelName != "tkll-model.h5":
		# 	continue
		model = models.load_model(modelPath + "\\" + modelName)
		file = open(
			"C:\\Users\\Admin\\Documents\\coding\\masterAI\\traffic sign detection\\logs\\validate.txt", "a")
		file.write(modelName + "\n")
		file.write("--------------\n")
		file.close()

		# NOT ROS Part
		# Parsing files in folder
		myPath = "C:\\Users\\Admin\\Documents\\coding\\masterAI\\traffic sign detection\\data2"
		onlyfiles = [f for f in listdir(myPath) if isfile(join(myPath, f))]
		for fileName in onlyfiles:
			result = fileName.split("_")
			myIndex = result[0]
			result = result[1]
			result = result[:-4]
			print(result)
			callbackFunction(myPath + "\\" + fileName,
							 expectedResult=result, index=myIndex)

		file = open(
			"C:\\Users\\Admin\\Documents\\coding\\masterAI\\traffic sign detection\\logs\\validate.txt", "a")
		file.write("\n")
		file.close()


if MODE == Mode.PIC:
	# NOT ROS Part
	# For 1 picture
	model = models.load_model(modelPath + "\\" + "model-150.h5")
	myPath = "C:\\Users\\Admin\\Documents\\coding\\masterAI\\traffic sign detection\\data2"
	onlyfiles = [f for f in listdir(myPath) if isfile(join(myPath, f))]
	expectedIndex = 2
	for fileName in onlyfiles:
		# print(fileName)
		result = fileName.split("_")
		myIndex = result[0]
		if myIndex != str(expectedIndex):
			continue
		result = result[1]
		result = result[:-4]
		print(result)
		callbackFunction(myPath + "\\" + fileName)
	cv2.waitKey(0)


# camera mode
if MODE == Mode.CAMERA:
	cam = cv2.VideoCapture(1)
	model = models.load_model(modelPath + "/" + "model-110.h5")

	while True:
		ret, frame = cam.read()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		# if frame == None:
		#	continue
		callbackFunctionVid(frame)

	cam.release()
	cv2.destroyAllWindows()
	pass


if MODE == Mode.VIDEO:
	vid = cv2.VideoCapture(videoPath)
	# frame
	currentframe = 0

	while(True):
		ret, frame = vid.read()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
		if ret:
			cv2.imshow("name", frame)
			currentframe += 1
		else:
			break
	vid.release()
	cv2.destroyAllWindows()
	pass
