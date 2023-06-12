import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from calendar import day_name, month
from operator import mod
from pyexpat import model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import *
from keras.optimizers import *
import numpy as np
import cv2
import random
from keras.models import *
import time
import math

dimImage = 128

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + 1.)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + 1.)
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
def dice(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)
    return score

# cnn = load_model("D:/container/AI_DCLV/goodModel/Model-CARLA11_GAZEBO_4-2-e15.hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})
cnn = load_model("D:/container/AI_DCLV/21-4/Model-S1_AUTO-2.hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})

def distance_between_lines(a1, b1, a2, b2):
    # Calculate slopes
    if a1 == a2:
        return abs(b1 - b2)
    # Calculate intersection point
    x0 = (b2 - b1) / (a1 - a2)
    y0 = a1 * x0 + b1
    # Calculate distance at intersection point
    y1 = a1 * x0 + b1 
    y2 = a2 * x0 + b2
    return abs(y1 - y2)

def isValidLane(X, Y, isLeftLane, width=128, minNumbersOfPoints=10):    
    if (len(X) < minNumbersOfPoints or len(Y) < minNumbersOfPoints):
        return False, None, None, None, None, None, None
    
    a, b = np.polyfit(X, Y, 1)        
    x1 = 0
    y1 = int(a*x1 + b)
    x2 = width - 1
    y2 = int(a*x2 + b)
    
    if (isLeftLane):
        if a >= 0:
            return False, None, None, None, None, None, None
    else:
        if a <= 0: 
            return False, None, None, None, None, None, None
        
    return True, x1, y1, x2, y2, a, b

def getIntersectionOfLines(coef1s, coef2s):
    # y = (a, b)
    # y = ax + b
    x0 = -int((coef1s[1] - coef2s[1])/(coef1s[0] - coef2s[0]))
    y0 = int(coef1s[0]*x0 + coef1s[1])
    return x0, y0

def removingOutliers(data1D, m = 2.0):
    d = np.abs(data1D - np.median(data1D))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return np.where(s < m)

def processing(img):    
    # Load binary image    
    height = img.shape[0]
    width = img.shape[1]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            cv2.drawContours(image=img, contours=[contour], contourIdx=-1, color=0, thickness=-1)

    # Find coordinates of all white pixels    
    yList, xList = np.where(threshold == 255)
    
    leftIdx = np.where(xList <= width//2)
    rightIdx = np.where(xList > width//2)

    yLeft = np.flip(yList[leftIdx])
    xLeft = np.flip(xList[leftIdx])
    _, indicesLeft = np.unique(yLeft, return_index=True)
    modifiedLeftY, modifiedLeftX = yLeft[indicesLeft], xLeft[indicesLeft]

    yRight = yList[rightIdx]
    xRight = xList[rightIdx]
    _, indicesRight = np.unique(yRight, return_index=True)
    modifiedRightY, modifiedRightX = yRight[indicesRight], xRight[indicesRight]
    
    # Remove outliers
    rmOutLierRight = removingOutliers(modifiedRightX)
    modifiedRightY = modifiedRightY[rmOutLierRight]
    modifiedRightX = modifiedRightX[rmOutLierRight]
    
    rmOutLierLeft = removingOutliers(modifiedLeftX)
    modifiedLeftY = modifiedLeftY[rmOutLierLeft]
    modifiedLeftX = modifiedLeftX[rmOutLierLeft]

    # emptyImage = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    # emptyImage[modifiedRightY, modifiedRightX] = (0, 0, 255)
    # emptyImage[modifiedLeftY, modifiedLeftX] = (0, 255, 0) 
    
    # y = a x + b
    copy_img = np.copy(img)
    
    isValidLeft, x1, y1, x2, y2, aL, bL = isValidLane(modifiedLeftX, modifiedLeftY, True)
    isValidRight, x3, y3, x4, y4, aR, bR = isValidLane(modifiedRightX, modifiedRightY, False)
    
    if (isValidLeft and isValidRight):
        xIntersection, yIntersection = getIntersectionOfLines((aL, bL), (aR, bR))
        if yIntersection < int(height*0.75) and xIntersection > 0.3*width and xIntersection < (1-0.3)*width:
            cv2.line(copy_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(copy_img, (x3, y3), (x4, y4), (0, 0, 255), 2)
            return [[[x1,y1],[x2,y2]],[[x3,y3],[x4,y4]]]
    return []

def runLaneDetectModel(input,save):
    frame = cv2.resize(input,(dimImage,dimImage))
    inputFrame = np.array([frame],dtype='float32')
    predict = cnn.predict(inputFrame,batch_size =1, verbose = 0)
    
    # =============== Backend ======================
    predict = predict[0]
    predict*=255
    cv2.imshow("predict",cv2.resize(np.array(predict,dtype='uint8'),(512,512)))
    outputImage = np.array(predict,dtype="uint8")
    _,outputImage = cv2.threshold(outputImage,127,255,cv2.THRESH_BINARY)
    listPoint = processing(outputImage)
    # =============== return output ======================
    return np.array(predict,dtype="uint8"),listPoint

def release_cap():
    pass
