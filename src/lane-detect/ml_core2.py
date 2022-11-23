import cv2
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt
import random

orgImgPath = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/image/"
labelImgPath = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/label/"
finalResultPath = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/finalResult/"


def getAreaBoundingBox(box):
    x, y, w, h = box
    return w * h


def load_images_from_folder(pathFolder):
    return os.listdir(pathFolder)


def processing(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 180, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    frameCopy = frame.copy()
    frameCopy = cv2.circle(frameCopy, (frameCopy.shape[0]//2,frameCopy.shape[1] - 10), radius=5, color=(255, 255, 255), thickness=-1)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    maxBoundRect = None
    maxBoundAreaRect = -1
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        currentArea = getAreaBoundingBox(boundRect[i])
        if currentArea > maxBoundAreaRect:
            maxBoundAreaRect = currentArea
            maxBoundRect = boundRect[i]

    # for i in range(len(contours)):
    #     color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    #     # cv2.drawContours(drawing, contours_poly, i, color)
    #     cv2.rectangle(frameCopy, (int(boundRect[i][0]), int(boundRect[i][1])), \
    #     (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)    

    if maxBoundAreaRect == -1:
        cv2.imwrite(finalResultPath + "err-" + str(time.time()) +
                    ".jpg", cv2.vconcat([frameCopy, frame]))
        return None
    else:
        x, y, w, h = int(maxBoundRect[0]), int(maxBoundRect[1]), int(
            maxBoundRect[2]), int(maxBoundRect[3])

        if w < 0.6*frame.shape[0] or \
                y + h < 0.85*frame.shape[1]:

            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0,256))        
            cv2.rectangle(frameCopy, (x, y), (x + w, y + h), color, 2)
            cv2.imwrite(finalResultPath + "err-" + str(time.time()
                        ) + ".jpg", cv2.vconcat([frameCopy, frame]))
            return None

        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0,256))
        cv2.rectangle(frameCopy, (x, y), (x + w, y + h), color, 2)

    cv2.imwrite(finalResultPath + str(time.time()) + \
                ".jpg", cv2.vconcat([frameCopy, frame]))

listOrgImages = load_images_from_folder(pathFolder=orgImgPath)
listLabelImages = load_images_from_folder(pathFolder=labelImgPath)

for singleLabelImage in listLabelImages:
    fullPath = labelImgPath + singleLabelImage
    frame = cv2.imread(fullPath)
    detectedRoad = processing(frame=frame)

    cv2.imshow("single label image", frame)
    if cv2.waitKey(25) == 27:
        break
