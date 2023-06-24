import os
from turtle import width
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
import copy

dimImage = 128



class stupidModel:
    def build():
        model = Sequential()
        model.add(Dense(512, activation = 'sigmoid', input_shape = (256,256,1)))
        model.add(Dense(512, activation = 'sigmoid'))
        return model

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
def randomImage(img,rotate, scale, Sx, Sy, isFlip):
	height, width = img.shape[:2]
	rotate_matrix = cv2.getRotationMatrix2D(center=(Sx+256,Sy+256), angle=rotate, scale=scale)
	rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))
	rotated_image = rotated_image[Sy:Sy+512,Sx:Sx+512]
	if isFlip:
		rotated_image = cv2.flip(rotated_image, 1)
	return cv2.resize(rotated_image,(128,128))

stpm = stupidModel()

# keras.losses.

cnn = stpm.unet(input_size=(dimImage,dimImage,1),loss=focal_tversky,metrics=dice)
# cnn = load_model("E:/DatasetContainer/AI_container/last/Model-CARLA2_Last.hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})

dataX = []
dataY = []

nameList = os.listdir("E:/DatasetContainer/LaneDetection/PROCESS_Kitty/extend_image/")

for eachName in nameList:
    tmpX = cv2.imread("E:/DatasetContainer/LaneDetection/PROCESS_Kitty/extend_image/" + eachName)
    tmpX = cv2.cvtColor(tmpX,cv2.COLOR_RGB2GRAY)
    dataX.append(np.array(tmpX))

    tmpY = cv2.imread("E:/DatasetContainer/LaneDetection/PROCESS_Kitty/extend_label/" + eachName)
    tmpY = cv2.cvtColor(tmpY,cv2.COLOR_RGB2GRAY)
    dataY.append(np.array(tmpY))
    # outImg = cv2.addWeighted(tmpY,0.5,tmpX,1,0)
    # cv2.imshow("tmp",outImg)
    # cv2.waitKey(200)

nameList = os.listdir("E:/DatasetContainer/AI_container/.labeled/image/")

testX = []
testY = []

for eachName in nameList:
    tmpX = cv2.imread("E:/DatasetContainer/AI_container/.labeled/image/" + eachName)
    tmpX = cv2.resize(tmpX,(dimImage,dimImage))
    tmpX = cv2.cvtColor(tmpX,cv2.COLOR_RGB2GRAY)
    testX.append(np.array(tmpX))

    tmpY = cv2.imread("E:/DatasetContainer/AI_container/.labeled/label/" + eachName)
    tmpY = cv2.resize(tmpY,(dimImage,dimImage))
    tmpY = cv2.cvtColor(tmpY,cv2.COLOR_RGB2GRAY)
    testY.append(np.array(tmpY))

# dataX = dataX[4:]
# dataY = dataY[4:]

dataX = np.array(dataX)
dataY = np.array(dataY)

testX = np.array(testX)
testY = np.array(testY)



dataY //= 255
testY //=255

testX  = testX.astype('float32')
testY  = testY.astype('float32')

width = 1366
height = 768
dataImage = []
dataLabel = []

for i in range(37,101):
    print(" ------------------- Epoch ",i," --------------------")

    dataImage = []
    dataLabel = []

    for eachData in range(1500):
        randParameter = [ 
        random.randint(0, 60)/2-15,
        random.randint(0, 20)/20+1,
        random.randint(0, width-512),
        random.randint(0, height-512),
        random.randint(0,1)==1]

        imageSource = random.randint(0, 336)
        dataImage.append(randomImage(dataX[imageSource],randParameter[0],randParameter[1],randParameter[2],randParameter[3],randParameter[4]))
        dataLabel.append(randomImage(dataY[imageSource],randParameter[0],randParameter[1],randParameter[2],randParameter[3],randParameter[4]))

    for eachData in range(0, 336):
        dataImage.append(cv2.resize(dataX[eachData],(dimImage,dimImage)))
        dataLabel.append(cv2.resize(dataY[eachData],(dimImage,dimImage)))

    dataImage = np.array(dataImage)
    dataLabel = np.array(dataLabel)

    dataImage  = dataImage.astype('float32')
    dataLabel  = dataLabel.astype('float32')  

    cnn.fit(dataImage, dataLabel, batch_size = 4, epochs = 1, validation_data=(testX, testY))
    if i%10 == 0:
        cnn.save("E:/DatasetContainer/AI_container/last/Model-CARLA2_epoch"+str(i)+".hdf5")
    cnn.save("E:/DatasetContainer/AI_container/last/Model-CARLA2_Last.hdf5")
