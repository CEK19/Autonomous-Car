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
cnn = load_model("D:/container/AI_DCLV/30-4/Model-S5_AUTO-15.hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})

def goodContour(rect):
    center,size,angle = rect
    mi,ma = min(size),max(size)
    if mi == 0:
        return 0
    if (ma/mi > 6 and mi*ma > 30):
        if (size[0] > size[1]):
            return 1
        else:
            return 2
    else:
        return 0
    
def getYVal(x):
    return x[1]

def processing(img):    
    # Load binary image
    cam = img.copy()
    cam = cv2.cvtColor(cam,cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    returnList = [[],[]]
    
    # Draw a bounding box around each contour
    for contour in contours:
        # Find the best fit rectangle for the contour
        rect = cv2.minAreaRect(contour)
        pos = -1
        # Draw the rectangle on the image
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        color = (0,255,0)
        if goodContour(rect) == 1:
            pos = 1 #right
            color = (255,100,0)
        elif goodContour(rect) == 2:
            pos = 0 #left
            color = (0,100,255)
        cv2.drawContours(cam, [box], 0, color, 2)

        endpoints = cv2.fitLine(box, cv2.DIST_L2, 0, 0.01, 0.01)
        # Calculate the endpoints of the line
        vx = endpoints[0]
        vy = endpoints[1]
        x0 = int(endpoints[2])
        y0 = int(endpoints[3])
        x1 = int(x0 - vx * 1000)
        y1 = int(y0 - vy * 1000)
        x2 = int(x0 + vx * 1000)
        y2 = int(y0 + vy * 1000)

        _,p1,p2 = cv2.clipLine((0,0,128,128),(x2, y2), (x1, y1))
        x1,y1 = p1
        x2,y2 = p2

        if (goodContour(rect)):
            cv2.line(cam, (x2, y2), (x1, y1), color, 2)

        if (pos != -1):
            if len(returnList[pos]) == 0:
                returnList[pos] = [[x1,y1],[x2,y2]]
                returnList[pos].sort(key=getYVal)
    # cv2.imshow("cam",cam)

    return returnList
    


def runLaneDetectModel(input,save):
    frame = cv2.resize(input,(dimImage,dimImage))
    inputFrame = np.array([frame],dtype='float32')
    predict = cnn.predict(inputFrame,batch_size =1, verbose = 0)
    # =============== Backend ======================
    predict = predict[0]
    predict*=255
    # cv2.imshow("predict",cv2.resize(np.array(predict,dtype='uint8'),(512,512)))
    outputImage = np.array(predict,dtype="uint8")
    listPoint = processing(outputImage)

    # =============== return output ======================
    return np.array(predict,dtype="uint8"),listPoint

def release_cap():
    pass
