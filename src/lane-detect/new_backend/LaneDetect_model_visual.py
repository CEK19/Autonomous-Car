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
cnn = load_model("D:/container/AI_DCLV/goodModel/Model-CARLA9_epoch2.hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (128*2,128))

def runLaneDetectModel(input):
    frame = cv2.resize(input,(dimImage,dimImage))
    inputFrame = np.array([frame],dtype='float32')
    predict = cnn.predict(inputFrame,batch_size =1, verbose = 0)
    # cv2.imshow("predict",np.array(predict[0],dtype='uint8'))
    # =============== Backend ======================
    predict = predict[0]
    predict*=255
    outputImage = np.array(predict,dtype="uint8")
    # outputImage *= 255
    cv2.imshow("output",outputImage)
    outputImage = cv2.adaptiveThreshold(outputImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    lines = cv2.HoughLines(outputImage, 1, np.pi / 180, 30, None, 0, 0)

    outputImage = cv2.cvtColor(outputImage,cv2.COLOR_GRAY2BGR)

    leftLaneData = []
    rightLaneData = []
    if lines is not None:
        for i in range(0, len(lines)):
            if lines[i][0][1] < 1:
                leftLaneData.append(lines[i][0])
            elif lines[i][0][1] < 2:
                pass
            else:
                rightLaneData.append(lines[i][0])
    else:
        print("line is none")

    ratio = -1
    if len(leftLaneData) != 0 and len(rightLaneData) != 0:
        leftLaneData = np.array(leftLaneData)
        leftLaneData = [np.mean(leftLaneData[:,0]),np.mean(leftLaneData[:,1])]
        rightLaneData = np.array(rightLaneData)
        rightLaneData = [np.mean(rightLaneData[:,0]),np.mean(rightLaneData[:,1])]
        # if max(leftLaneData[2],rightLaneData[2]) < 20 and max((leftLaneData[3],rightLaneData[3])) <  0.05:
        for eachLane in [leftLaneData,rightLaneData]:
            print("each lane:",eachLane)
            rho = eachLane[0]
            theta = eachLane[1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            x = (64 - y0)/a
            x = x0 -b*x
            d1 = rho/a
            d2 = 128*math.tan(theta)
            if eachLane == leftLaneData:
                ld = d2-d1+128/2
            else:
                rd = d1+d2-128/2
                ratio = ld/(ld+rd)
            if theta < 1:
                cv2.line(outputImage, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
                outputImage = cv2.putText(outputImage,str(int(d1))+" "+str(int(d2)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            elif theta < 2:
                cv2.line(outputImage, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)
            else:
                cv2.line(outputImage, pt1, pt2, (255,0,0), 3, cv2.LINE_AA)
                outputImage = cv2.putText(outputImage,str(int(d1))+" "+str(int(d2)),(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.imshow("output",outputImage)
    else:
        print("Lane unknow")
    # =============== return output ======================

    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    frame = cv2.hconcat([frame,outputImage])
    out.write(frame)

    return ratio

def release_cap():
    out.release()

