import cv2
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt
import random

orgImgPath = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/image/"
labelImgPath = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/label/"

# preprocessingPath = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/preprocessing/"
# visualizeResultPath = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/finalResult/"
# timeCompare = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/timeCompare/"
grayScale_1 = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/1-grayscale/"
gaussCanny_2 = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/2-gauss-canny/"
threshHold_3 =  "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/3-threshhold/"
countour_4 = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/4-countour/"
maxCountour_5 =  "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/5-maxcontour/"
cropImg_5_5 = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/5.5-crop-img/"
masking_6 = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/6-masking/"
regression_7 = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/src/lane-detect/fullOutput/7-regression/"
currentLabel = ""


def getIntersectionOfLines(coef1s, coef2s):
    # y = (a, b)
    # y = ax + b
    x0 = -int((coef1s[1] - coef2s[1])/(coef1s[0] - coef2s[0]))
    y0 = int(coef1s[0]*x0 + coef1s[1])
    return x0, y0


def getAreaBoundingBox(box):
    x, y, w, h = box
    return w * h


def load_images_from_folder(pathFolder):
    return os.listdir(pathFolder)


def findingAngleInRadian(startPoint, endPoint):  # (x,y), (x,y)
    return math.atan((endPoint[0] - startPoint[0])/(endPoint[1] - startPoint[1]))

# https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
# https://codereview.stackexchange.com/questions/230928/remove-outliers-from-n-dimensional-data
def removingOutliners(data1D, m = 5.0):
    d = np.abs(data1D - np.median(data1D))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return np.where(s < m)

def algorithm(preprocessImage, centerPoint):
    writeImage(pathFolder=cropImg_5_5, image=preprocessImage, prefix=currentLabel)
    col = preprocessImage.shape[1]
    row = preprocessImage.shape[0]
    centerX = col//2
    cdstP = preprocessImage.copy()

    # remove not necessary pont in the top of image
    deltaYFromTop = row//10
    # remove not necessary point in middle of lane (replace middle with black polygon):
    cdstP[0: deltaYFromTop, :] = 0
    deltaXFromCenter = 5  # always > 0
    startNonNecessaryL = (col//9, row)
    endNonNecessaryL = (centerX - deltaXFromCenter, row//2)

    startNonNecessaryR = (col - col//9, row)
    endNonNecessaryR = (centerX + deltaXFromCenter, row//2)

    pts = np.array([
        startNonNecessaryL, endNonNecessaryL,
        endNonNecessaryR, startNonNecessaryR,
    ], np.int32)

    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(cdstP, pts=[pts], color=0)
    writeImage(pathFolder=masking_6, image=cdstP, prefix=currentLabel)

    positionYs, positionXs = np.where(cdstP == 255)

    # find best fit line left points in images
    leftCondition = np.where(positionXs < col//2)    
    rightCondition = np.where(positionXs >= col//2)
        
    arrayLeftX, arrayLeftY = positionXs[leftCondition], positionYs[leftCondition]
    
    # find best fit line right points in images
    arrayRightX, arrayRightY = positionXs[rightCondition], positionYs[rightCondition]
    

    if (len(arrayLeftX) <= 10 or len(arrayRightX) <=10):
        return -1, None, None, None, None
    
    rmOutLinerConditionLeft  = removingOutliners(arrayLeftX) 
    arrayLeftX, arrayLeftY = arrayLeftX[rmOutLinerConditionLeft], arrayLeftY[rmOutLinerConditionLeft]    
    
    rmOutLinerConditionRight = removingOutliners(arrayRightX) 
    arrayRightX, arrayRightY = arrayRightX[rmOutLinerConditionRight], arrayRightY[rmOutLinerConditionRight]    

    # y = a x + b
    aL, bL = np.polyfit(arrayLeftX, arrayLeftY, 1)
    aR, bR = np.polyfit(arrayRightX, arrayRightY, 1)

    fromLeftX, fromLeftY = getIntersectionOfLines(
        coef1s=(aL, bL), coef2s=(0, row))
    targetLeftX, targetLeftY = getIntersectionOfLines(
        coef1s=(aL, bL), coef2s=(0, 0))

    fromRightX, fromRightY = getIntersectionOfLines(
        coef1s=(aR, bR), coef2s=(0, row))
    targetRightX, targetRightY = getIntersectionOfLines(
        coef1s=(aR, bR), coef2s=(0, 0))

    cdstP = cv2.line(cdstP, (fromLeftX, fromLeftY),
                     (targetLeftX, targetLeftY), color=127, thickness=2)
    cdstP = cv2.line(cdstP, (fromRightX, fromRightY),
                     (targetRightX, targetRightY), color=127, thickness=2)

    xMin, xMax = 0, col
    if (fromLeftX < 0):
        xMin = fromLeftX
    if (fromRightX > col):
        xMax = fromRightX

    plt.xlim([xMin, xMax])
    plt.plot([fromLeftX, targetLeftX], [fromLeftY, targetLeftY], color='red', linewidth=3)
    plt.plot([fromRightX, targetRightX],  [fromRightY, targetRightY], color='red', linewidth=3)
    plt.imshow(cdstP)
    plt.savefig(regression_7 + "final" + currentLabel  +str(time.time()) + ".jpg", bbox_inches='tight')
    plt.clf()

    return centerPoint[0]/abs(xMax - xMin), aL, bL, aR, bR


def writeImage(pathFolder, image, prefix=""):
    cv2.imwrite(pathFolder + prefix + str(time.time()) +".jpg", image)    
    

def preprocessing(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    writeImage(pathFolder=grayScale_1, image=gray, prefix=currentLabel)

    gaussianBlur = cv2.GaussianBlur(gray, (5, 5), 0)
    cannyEdgeDetection = cv2.Canny(gaussianBlur, 100, 200)
    writeImage(pathFolder=gaussCanny_2, image=cannyEdgeDetection, prefix=currentLabel)

    _, thresh = cv2.threshold(gray, 100, 180, cv2.THRESH_BINARY)
    writeImage(pathFolder=threshHold_3, image=thresh, prefix=currentLabel)
    
    contours, _ = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    frameCopy = cannyEdgeDetection.copy()

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    noMeaningValue = np.full((frame.shape[1], frame.shape[0]), None)

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
        # cv2.imwrite(preprocessingPath + "err-" + str(time.time()) +
        #             ".jpg", cv2.vconcat([frameCopy, cannyEdgeDetection]))
        writeImage(pathFolder=maxCountour_5, image=frameCopy, prefix="err" + currentLabel)
        return noMeaningValue, None, None, None
    else:
        x, y, w, h = int(maxBoundRect[0]), int(maxBoundRect[1]), int(
            maxBoundRect[2]), int(maxBoundRect[3])
        
        centerPoint = (frameCopy.shape[0]//2, 0)
        if w < 0.6*frame.shape[0] or \
                y + h < 0.85*frame.shape[1]:
            cv2.rectangle(frameCopy, (x, y), (x + w, y + h), 255, 2)
            writeImage(pathFolder=maxCountour_5, image=frameCopy, prefix="err" + currentLabel)
            return noMeaningValue, None, None, None

        prevFrame = frameCopy.copy()
        cv2.rectangle(frameCopy, (x, y), (x + w, y + h), 255, 2)
        writeImage(pathFolder=maxCountour_5, image=frameCopy, prefix=currentLabel)
                
        return prevFrame[y: y + h, x: x + w], centerPoint, x, y


# listOrgImages = load_images_from_folder(pathFolder=orgImgPath)
listLabelImages = sorted(load_images_from_folder(pathFolder=labelImgPath))
# file = open("nhan-backend.txt", 'a')
# file = open(timeCompare + "nhan-time.txt", 'a')
# valid = [53, 59, 60, 61, 65, 74, 81, 94, 172, 179, 215]
for idx, singleLabelImage in enumerate(listLabelImages):
    # if (idx not in valid):
    #     continue
    currentLabel = singleLabelImage
    fullPath = labelImgPath + singleLabelImage
    frame = cv2.imread(fullPath)
    
    startTime = time.time()    
    detectedRoad, centerPoint, x, y = preprocessing(frame=frame)
    
    ratio = -1    
    aL, bL, aR, bR = 0, 0 , 0 , 0
    if (detectedRoad.any() == None):
        pass
    else:
        ratio, aL, bL, aR, bR = algorithm(detectedRoad, centerPoint=centerPoint)        
        
    endTime = time.time()        
    # file.write(f'{endTime - startTime}\n')

    # if (ratio == -1):
    #     file.write(f'{singleLabelImage}: left: invalid, right: invalid, x: invalud, y: invalid\n')    
    # else:
    #     file.write(f'{singleLabelImage}: left: y={aL}x+{bL}, right: y={aR}x+{bR}, x:{x}, y:{y}\n')    
    # file.write(f'{singleLabelImage}: {ratio}\n')
    # cv2.imshow("single label image", frame)
    if cv2.waitKey(25) == 27:
        break
# file.close()
