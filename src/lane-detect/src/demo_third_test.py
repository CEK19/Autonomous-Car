import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from calendar import day_name, month
from operator import mod
from pyexpat import model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import keras.losses
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import *
from keras.optimizers import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from PIL import Image as im
import random
from keras.models import *
from enum import Enum
import sys
import datetime

dimImage = 128


class stupidModel:

    def unet(self, input_size, loss, metrics):
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        
        self.model = Model(inputs = inputs, outputs = conv10)
        self.model.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics =[metrics])
        return self.model

smooth=1
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
def dice(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
stpm = stupidModel()

# keras.losses.

# cnn = stpm.unet(input_size=(dimImage,dimImage,1),loss=focal_tversky,metrics=dice)

dataX = []
dataY = []

pathTest = "D:/container/AI_DCLV/readData/gazebo/image/"
pathOutput = "D:/container/AI_DCLV/readData/gazebo/output/"

nameList = os.listdir(pathTest)

# maxImage = 2
# if (len(nameList) > maxImage):
#     nameList = nameList[0:maxImage]

for eachName in nameList:
    tmpX = cv2.imread(pathTest + eachName)
    tmpX = cv2.resize(tmpX,(dimImage,dimImage))
    tmpX = cv2.cvtColor(tmpX,cv2.COLOR_RGB2GRAY)
    dataX.append(np.array(tmpX))

    # tmpY = cv2.imread("E:/DatasetContainer/LaneDetection/PROCESS_Kitty/label/" + eachName)
    # tmpY = cv2.resize(tmpY,(dimImage,dimImage))
    # tmpY = cv2.cvtColor(tmpY,cv2.COLOR_RGB2GRAY)
    # dataY.append(np.array(tmpY))

# for tmp_modelCounter in range(21,40):

# dataX = dataX[0:128]
dataX = np.array(dataX)
dataX  = dataX.astype('float32')

modelCounter = 27
print(len(dataX))
# cnn = load_model("E:/DatasetContainer/AI_container/last/Model-CARLA2_epoch"+str(modelCounter)+".hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})
cnn = load_model("D:/container/AI_DCLV/gazebo/Model-CARLA11_GAZEBO_4-1-e7.hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})

predict = cnn.predict(dataX,batch_size =8)
tmp = np.zeros((dimImage,dimImage,3))
nameTestList = os.listdir(pathTest)
# predict*=255
for each in range(0,len(dataX)):
    # cv2.imwrite(pathOutput.replace("output_testFolder","output_testFolder_4label")+nameTestList[each],cv2.resize(predict[each]*255,(512,512)))
    tmp = cv2.cvtColor(predict[each],cv2.COLOR_GRAY2RGB)
    tmp = tmp*255
    tmp[:,:,1] = 0
    


    outImg = cv2.resize(tmp,(dimImage,dimImage))

    orgImg = cv2.imread(pathTest+nameTestList[each])
    orgImg = cv2.resize(orgImg,(dimImage,dimImage))

    outImg = np.array(outImg, dtype = np.uint8)
    orgImg = np.array(orgImg, dtype = np.uint8)
    
    outImg = cv2.addWeighted(outImg,0.5,orgImg,1,0)
    # cv2.imshow("image",outImg)
    cv2.imwrite(pathOutput + "predicOf_"+nameTestList[each],outImg)
    print("save",nameTestList[each])
    tmp = np.zeros((dimImage,dimImage,3))
    # break