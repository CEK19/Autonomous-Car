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
        # self.model.compile(optimizer = Adam(lr = 3e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
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

def ImageArgument(frame,brightnessOffset,noise,blur):
    frame = np.array(frame,dtype="int")
    frame = np.clip(frame - brightnessOffset,0,255)
    for j in range(int(100*noise)):
        frame[int(random.random()*128)][int(random.random()*128)] = random.random()*255
    frame = np.array(frame,dtype="uint8")
    frame = cv2.blur(frame, (blur,blur)) 
    return frame

def randomize_rect(image, x, y, width, height):
    # Tạo một bản sao của hình ảnh đầu vào để tránh ảnh hưởng đến hình ảnh gốc
    img_copy = np.copy(image)

    # Tạo một hình chữ nhật ngẫu nhiên với kích thước được xác định bởi đầu vào (x,y,width,height)
    rect = img_copy[y:y+height, x:x+width]

    # Tạo các giá trị ngẫu nhiên cho các pixel trong hình chữ nhật
    random_values = np.random.randint(low=0, high=256, size=rect.shape)

    # Gán các giá trị ngẫu nhiên vào các pixel trong hình chữ nhật
    rect[:] = random_values

    # Trả về hình ảnh được cập nhật
    return img_copy

def augment_image(image):
    # Randomly apply Gaussian blur
    blur_radius = random.randint(0, 3)
    if blur_radius > 0:
        image = cv2.GaussianBlur(image, (2 * blur_radius + 1, 2 * blur_radius + 1), 0)

    # Randomly adjust brightness
    brightness = random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

    dark = random.randint(-150, 10)
    lut = np.arange(0, 256, 1, dtype=np.int32)
    lut = np.clip(lut + dark, 0, 255).astype(np.uint8)
    image = cv2.LUT(image, lut)

    # Randomly add frost effect
    frost_level = random.uniform(0, 0.5)
    noise = np.random.randn(*image.shape) * 50
    noise = np.clip(noise, -255, 255).astype(np.uint8)
    image = cv2.addWeighted(image, 1 - frost_level, noise, frost_level, 0)

    # Randomly hide some part of the image
    x,y,w,h = np.random.randint(80),np.random.randint(80),np.random.randint(47),np.random.randint(47)
    image = randomize_rect(image,x,y,w,h)
    return image,[x,y,w,h]

stpm = stupidModel()

# keras.losses.

# cnn = stpm.unet(input_size=(dimImage,dimImage,1),loss=focal_tversky,metrics=dice)
cnn = load_model("D:/container/AI_DCLV/10-4/Model-17_RAS-4.hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})

dataX = []
dataY = []

# dataLink = "D:/container/AI_DCLV/readData/labeled_v6/"
dataLink = "D:/container/AI_DCLV/readData/realDataGenerator/generatedData/"

nameList = os.listdir(dataLink+"image")

print("Start read data")

for eachName in nameList:
    tmpX = cv2.imread(dataLink+"image/" + eachName)
    tmpX = cv2.resize(tmpX,(dimImage,dimImage))
    tmpX = cv2.cvtColor(tmpX,cv2.COLOR_RGB2GRAY)
    dataX.append(np.array(tmpX))
    tmpX = cv2.flip(tmpX,1)
    dataX.append(np.array(tmpX))

    tmpY = cv2.imread(dataLink+"label/" + eachName)
    tmpY = cv2.resize(tmpY,(dimImage,dimImage))
    tmpY = cv2.cvtColor(tmpY,cv2.COLOR_RGB2GRAY)
    _,tmpY = cv2.threshold(tmpY,1,255,cv2.THRESH_BINARY)
    dataY.append(np.array(tmpY))
    tmpY = cv2.flip(tmpY,1)
    dataY.append(np.array(tmpY))

print("Start train")

# dataX  = np.array(dataX)
# dataY  = np.array(dataY)

print(" ------------------- Start train --------------------")
for j in range(len(dataX)):
    dataX[j],rectPos = augment_image(dataX[j])
    dataY[j] = cv2.rectangle(dataY[j],(rectPos[0],rectPos[1]),(rectPos[0]+rectPos[2],rectPos[1]+rectPos[3]),0,-1)
dataX = np.array(dataX,dtype="float32")
dataY = np.array(dataY,dtype="float32")

dataY /= 255
print(" Start train !")
history = cnn.fit(dataX, dataY, batch_size = 2, epochs = 64)
cnn.save("D:/container/AI_DCLV/10-4/Model-17_RAS-5.hdf5")
print(history)
