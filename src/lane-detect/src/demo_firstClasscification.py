from pyexpat import model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from PIL import Image as im


class ShallowNet:
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(128, (3, 3), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

class MiniVGGNet:
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1


        model.add(Conv2D(64, (3, 3), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))

        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))

        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))

        return model

def getFrameFromvideo(videoLink):
    cap = cv2.VideoCapture(videoLink)
    rtv = []
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame,(64,64))
            rtv.append(np.array(frame))
        else: 
            break
    cap.release()
    print(len(rtv))
    return rtv

def int2name(test, stt):
    header = "D:/NDT/PY_ws/DCLV/"
    if stt > 94:
        if test:
            header += "gt_image_2/umm_road_0000"
        else:
            header += "image_2/umm_0000"
        stt -= 94
    else:
        if test:
            header += "gt_image_2/um_road_0000"
        else:
            header += "image_2/um_0000"

    if (stt < 10):
        header += "0"
    header += str(stt)
    header += ".png"
    return header

# ---------------------------------------------------------------------------------
print(" =========== START =============")

train_X = []
train_Y = []

for sttImage in range(0,150):
    inimg = cv2.imread(int2name(False,sttImage))
    inimg = inimg[0:370,415:785]
    inimg = cv2.cvtColor(inimg,cv2.COLOR_RGB2GRAY)
    inimg = cv2.resize(inimg,(64,64))
    # cv2.imshow("train",inimg)
    train_X.append(np.array(inimg))


    inimg = cv2.imread(int2name(True,sttImage))
    inimg = inimg[0:370,415:785]
    inimg = cv2.cvtColor(inimg,cv2.COLOR_RGB2GRAY)
    ret,inimg = cv2.threshold(inimg, 50, 255, cv2.THRESH_BINARY)
    inimg = cv2.resize(inimg,(64,64))
    # cv2.imshow("test",inimg)
    # cv2.waitKey(500)
    inimg = np.reshape(inimg,(64*64))
    train_Y.append(np.array(inimg))

# test_X = np.array(train_X[100:150])
# test_Y = np.array(train_Y[100:150])
train_X = np.array(train_X)
train_Y = np.array(train_Y)

cnn = MiniVGGNet.build(64,64,1,64*64)
opt = SGD(lr=0.005)
cnn.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# cnn.save("D:/NDT/PY_ws/DCLV/Output/basicCNN/basic.hdf5")



for i in range(0,100):
    H = cnn.fit(train_X, train_Y, validation_data=(train_X, train_Y),batch_size=32, epochs=1, verbose=1)
    predict = cnn.predict(train_X,batch_size=32)
    exampleMatrix = np.reshape(predict[1],(64,64))
    exampleMatrix = np.asmatrix(exampleMatrix)
    cv2.imwrite("D:/NDT/PY_ws/DCLV/Output/basicCNN/"+str(i)+".png",cv2.resize(np.array(exampleMatrix),(512,512)))
