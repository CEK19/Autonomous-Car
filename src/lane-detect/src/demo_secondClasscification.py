from calendar import day_name, month
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
import random
from keras.models import load_model
from enum import Enum
import sys
import datetime



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


        model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

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
    if stt > 189:
        if test:
            header += "gt_image_2/uu_road_0000"
        else:
            header += "image_2/uu_0000"
        stt -= 189
    elif stt > 94:
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

def increaseColor(inColor):
    if (inColor + 64) > 255:
        return 255
    else:
         return (inColor + 64)

def increaseColorZone(inimage,x,y,w,h,c):
    for i in range(x,x+w):
        for j in range(y,y+h):
            inimage[i][j][c] = increaseColor(inimage[i][j][c])

class runningType(Enum):
    createNewAI = 1
    checkLatestAI = 2
    checkSpecificAI = 3


# ---------------------------------------------------------------------------------
print(" =========== CONFIG =============")
thisRunningType = runningType.checkLatestAI
# D:\NDT\PY_ws\DCLV\src\AI_containter\latestAI_in4.txt
boxSize = 32
outputSize = 16
timeData = datetime.datetime.now()

print(" =========== START =============")

dataListX = []
dataListY = []

random.seed(5)

for sttImage in range(0,286):
    inimg = cv2.imread(int2name(False,sttImage))
    # inimg = inimg[0:370,415:785]
    # inimg = cv2.cvtColor(inimg,cv2.COLOR_RGB2GRAY)
    dataListX.append(np.array(inimg))

    inimg = cv2.imread(int2name(True,sttImage))
    # inimg = inimg[0:370,415:785]
    # size: 370:370
    inimg = cv2.cvtColor(inimg,cv2.COLOR_RGB2GRAY)
    ret,inimg = cv2.threshold(inimg, 50, 255, cv2.THRESH_BINARY)
    dataListY.append(np.array(inimg))


# ======================= CREATE NEW AI =======================
if thisRunningType == runningType.createNewAI:
    print(" =========== CREATE NEW AI =============")
    cnn = MiniVGGNet.build(boxSize,boxSize,3,3)
    opt = SGD(lr=0.005)
    cnn.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    timeString = str(timeData.year)+"-"+str(timeData.month)+"-"+str(timeData.day)+"-"+str(timeData.hour)+"h"+str(timeData.minute)

    AIversion = 0
    allAI_in4 = ""
    with open('DCLV/src/AI_containter/allAI_in4.txt', 'r') as f:
        allAI_in4 = f.read()
        allAI_in4 = allAI_in4.split("\n---")
        AIversion = int(allAI_in4[0]) + 1
    
    with open('DCLV/src/AI_containter/latestAI_in4.txt', 'w') as f:
        f.write("t1_v"+str(AIversion)+"_"+timeString+"_Final.hdf5\n")
        f.write("boxSize: "+str(boxSize)+"\n")
        f.write("outputSize: "+str(outputSize)+"\n")
        f.write("numberLabel: "+str(3))

    for eachEpoch in range(0,10):
        print(" ======= EPOCHS ",eachEpoch," ============")
        train_X = []
        train_Y = []

        numLaneLabel = 0

        i=0

        while (i<10000):
            if (i < 1000):
                imgid = int(random.random()*20)
            else:
                imgid = int(random.random()*266)+20  
            cx = int(random.random()*(len(dataListX[imgid])-boxSize))
            cy = int(random.random()*(len(dataListX[imgid][0])-boxSize))
            train_X.append(np.array(dataListX[imgid][cx:cx+boxSize,cy:cy+boxSize]))
            ansImage = np.array(dataListY[imgid][int(cx+(boxSize - outputSize)/2):int(cx+(boxSize + outputSize)/2),int(cy+(boxSize - outputSize)/2):int(cy+(boxSize + outputSize)/2)])
            ansImage = np.reshape(ansImage,((outputSize)*(outputSize)))
            setAnsImage = set(ansImage)
            if 0 in setAnsImage:
                # no lane in this image
                if 255 in setAnsImage:
                    train_Y.append(np.array([0,0,1]))
                    # numLaneLabel+=1
                else:
                    train_Y.append(np.array([1,0,0]))
            else:
                # lane in this image
                train_Y.append(np.array([1,0,0]))

            if (train_Y[-1][0] == 1):
                if (random.random() < 0.975):
                    train_X.pop()
                    train_Y.pop()
                else:
                    i+=1
            else:
                i+=1

        test_X = np.array(train_X[0:1000])
        test_Y = np.array(train_Y[0:1000])
        train_X = np.array(train_X[1000:len(train_X)])
        train_Y = np.array(train_Y[1000:len(train_Y)])

        print("Num lane lable: ",numLaneLabel)
        print("size of dataset: ",len(train_X) + 1000)


        H = cnn.fit(train_X, train_Y, validation_data=(test_X, test_Y),batch_size=32, epochs=1, verbose=1)
        cnn.save("D:/NDT/PY_ws/DCLV/src/AI_containter/t1_v"+str(AIversion)+"_"+timeString+"_e" + str(eachEpoch) +".hdf5")
    
    cnn.save("D:/NDT/PY_ws/DCLV/src/AI_containter/t1_v"+str(AIversion)+"_"+timeString+"_Final.hdf5")
    allAI_in4[0] = str(AIversion)
    newIn4 = "t1_v"+str(AIversion)+"_"+timeString+"_Final.hdf5\nboxSize: "+str(boxSize) + "\noutputSize: "+str(outputSize)+"\nnumberLabel: "+str(3)
    allAI_in4.append(newIn4)
    allAI_in4 = "\n---".join(allAI_in4)
    with open('DCLV/src/AI_containter/allAI_in4.txt', 'w') as f:
        f.write(allAI_in4)

# ======================= CHECK LATEST AI =======================
if thisRunningType == runningType.checkLatestAI:
    allAI_in4 = ""
    with open('DCLV/src/AI_containter/latestAI_in4.txt', 'r') as f:
        allAI_in4 = f.read()
        allAI_in4 = allAI_in4.split("\n")
    cnn = load_model("D:/NDT/PY_ws/DCLV/src/AI_containter/"+allAI_in4[0])
    boxSize = int(allAI_in4[1].split(" ")[1])
    outputSize = int(allAI_in4[2].split(" ")[1])
    for tmp4loop in range(20,40):
        print("Test Image ",tmp4loop)
        # testImg = cv2.imread("D:/NDT/PY_ws/DCLV/testcase/test" + str(tmp4loop) + ".png")
        testImg = cv2.imread(int2name(False,tmp4loop))
        # testImg = testImg[0:370,200:1000]
        # testImg = cv2.cvtColor(testImg,cv2.COLOR_RGB2GRAY)

        test_X = []

        for i in range(0,int((len(testImg)-boxSize)/outputSize)):
            for  j in range(0,int((len(testImg[0])-boxSize)/outputSize)):
                test_X.append(np.array(testImg[i*outputSize:i*outputSize+boxSize,j*outputSize:j*outputSize+boxSize]))
                # if i==0:
                #     print(i*(boxSize-16),(i+1)*(boxSize-16)+16,j*(boxSize-16),(j+1)*(boxSize-16)+16)
                #     print("just append ",test_X[len(test_X)-1].shape)
        test_X = np.array(test_X)
        predict = cnn.predict(test_X,batch_size=32)
        testImg = cv2.imread(int2name(False,tmp4loop))
        # testImg = testImg[0:370,200:1000]
        # testImg = cv2.imread("D:/NDT/PY_ws/DCLV/testcase/test" + str(tmp4loop) + ".png")

        i=0
        j=0
                
        for i in range(0,int((len(testImg)-boxSize)/outputSize)):
            for  j in range(0,int((len(testImg[0])-boxSize)/outputSize)):
                each = i*int((len(testImg[0])-boxSize)/outputSize)+j
                for counterLabel in range(0,3):
                    if predict[each][counterLabel] > 0.7:
                        increaseColorZone(testImg,i*outputSize+12,j*outputSize+12,outputSize,outputSize,counterLabel)
                

        cv2.imwrite("D:/NDT/PY_ws/DCLV/Output/output-cnn-5-6/test-"+str(tmp4loop)+".png",testImg)

# ======================= CHECK SPECIFIC AI =======================
if thisRunningType == runningType.checkSpecificAI:
    AIname = ""
    cnn = load_model("D:/NDT/PY_ws/DCLV/src/AI_containter/"+AIname)
    # boxSize = ?
    # outputSize = ?
    for tmp4loop in range(20,40):
        print("Test Image ",tmp4loop)
        # testImg = cv2.imread("D:/NDT/PY_ws/DCLV/testcase/test" + str(tmp4loop) + ".png")
        testImg = cv2.imread(int2name(False,tmp4loop))
        # testImg = testImg[0:370,200:1000]
        # testImg = cv2.cvtColor(testImg,cv2.COLOR_RGB2GRAY)

        test_X = []

        for i in range(0,int((len(testImg)-boxSize)/outputSize)):
            for  j in range(0,int((len(testImg[0])-boxSize)/outputSize)):
                test_X.append(np.array(testImg[i*outputSize:i*outputSize+boxSize,j*outputSize:j*outputSize+boxSize]))
                # if i==0:
                #     print(i*(boxSize-16),(i+1)*(boxSize-16)+16,j*(boxSize-16),(j+1)*(boxSize-16)+16)
                #     print("just append ",test_X[len(test_X)-1].shape)
        test_X = np.array(test_X)
        predict = cnn.predict(test_X,batch_size=32)
        testImg = cv2.imread(int2name(False,tmp4loop))
        # testImg = testImg[0:370,200:1000]
        # testImg = cv2.imread("D:/NDT/PY_ws/DCLV/testcase/test" + str(tmp4loop) + ".png")

        i=0
        j=0
                
        for i in range(0,int((len(testImg)-boxSize)/outputSize)):
            for  j in range(0,int((len(testImg[0])-boxSize)/outputSize)):
                each = i*int((len(testImg[0])-boxSize)/outputSize)+j
                for counterLabel in range(0,3):
                    if predict[each][counterLabel] > 0.7:
                        increaseColorZone(testImg,i*outputSize+12,j*outputSize+12,outputSize,outputSize,counterLabel)
                

        cv2.imwrite("D:/NDT/PY_ws/DCLV/Output/output-cnn-5-6/test-"+str(tmp4loop)+".png",testImg)

