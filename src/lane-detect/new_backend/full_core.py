import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
videoPath = "D:/container/AI_DCLV/readData/video/angle7.MOV"
outputPath = "D:/NDT/PY_ws/DCLV/backend/ans/"


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

# cnn = load_model("D:/container/AI_DCLV/goodModel/Model-CARLA9_epoch2.hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})
cnn = load_model("D:/container/AI_DCLV/gazebo/Model-CARLA11_GAZEBO_4-1-e7.hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})

counter = 1

dataX = []
dataY = []
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
# Capture frame-by-frame
    # ret, frame = cap.read()
    frame = cv2.imread("D:/NDT/PY_ws/DCLV/ROS_zone/history/"+str(counter)+".png")
    # print(len(frame))
    ret = True
    if ret == True:
        # cv2.imwrite(outputPath+"org.png",frame)
        # =============== Preprocess and run predic ======================
        
        frame = cv2.resize(frame,(dimImage,dimImage))
        inputFrame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        # cv2.imwrite(outputPath+"preprocess.png",inputFrame)
        inputFrame = np.array([inputFrame],dtype='float32')
        startTime = time.time()
        predict = cnn.predict(inputFrame,batch_size =1, verbose = 0)
        avgTime = time.time() - startTime
        # =============== Backend ======================
        
        outputImage = cv2.cvtColor(predict[0],cv2.COLOR_GRAY2BGR)
        outputImage = outputImage*255
        cv2.imshow("predic",outputImage)
        # cv2.imwrite(outputPath+"output.png",outputImage)
        outputImage = cv2.cvtColor(outputImage,cv2.COLOR_BGR2GRAY)
        # print(outputImage.dtype)
        outputImage = np.array(outputImage,dtype="uint8")
        outputImage = cv2.adaptiveThreshold(outputImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        lines = cv2.HoughLines(outputImage, 1, np.pi / 180, 50, None, 0, 0)
        
        

        
        leftLaneData = []
        rightLaneData = []
            
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                # t = (512-rho*b)/a
                
                if lines[i][0][1] < 1:
                    # cv2.line(frame, pt1, pt2, (50,0,255), 1, cv2.LINE_AA)
                    # leftLaneData.append([x0 + t*(-b),lines[i][0][1]])
                    leftLaneData.append(lines[i][0])
                elif lines[i][0][1] < 2:
                    # cv2.line(frame, pt1, pt2, (0,100,0), 1, cv2.LINE_AA)
                    pass
                else:
                    # cv2.line(frame, pt1, pt2, (255,0,50), 1, cv2.LINE_AA)
                    # rightLaneData.append([x0 + t*(-b),lines[i][0][1]])
                    rightLaneData.append(lines[i][0])

        # blankImage = np.zeros((128,128,3),dtype="uint8")
        # frame = cv2.hconcat([blankImage,frame,blankImage])
        # blankImage = np.zeros((128,128*3,3),dtype="uint8")
        # frame = cv2.vconcat([frame,blankImage])

        if len(leftLaneData) != 0 and len(rightLaneData) != 0:

            leftLaneData = np.array(leftLaneData)
            leftLaneData = [np.mean(leftLaneData[:,0]),np.mean(leftLaneData[:,1])]

            rightLaneData = np.array(rightLaneData)
            rightLaneData = [np.mean(rightLaneData[:,0]),np.mean(rightLaneData[:,1])]
            ratio = 0
            for eachLane in [leftLaneData,rightLaneData]:
                rho = eachLane[0]
                theta = eachLane[1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

                d1 = rho/a
                d2 = 128*math.tan(theta)
                if eachLane == leftLaneData:
                    # print("l")
                    ld = d2-d1+128/2
                else:
                    # print("r")
                    rd = d1+d2-128/2
                    ratio = ld/(ld+rd)
                if theta < 1:
                    cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
                    frame = cv2.putText(frame,str(int(d1))+" "+str(int(d2)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                elif theta < 2:
                    cv2.line(frame, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)
                else:
                    cv2.line(frame, pt1, pt2, (255,0,0), 3, cv2.LINE_AA)
                    frame = cv2.putText(frame,str(int(d1))+" "+str(int(d2)),(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        avgTime = time.time() - startTime

        # outputImage = cv2.putText(outputImage, "t: "+str(avgTime)[:6]+" ms", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        # =============== Show output ======================
        counter += 1
        print("frame",counter,"done in",avgTime*1000,"ms and ratio",ratio)
        frame = cv2.resize(frame,(512,512))
        frame = cv2.putText(frame,str(ld),(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        frame = cv2.putText(frame,str(rd),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        frame = cv2.putText(frame,str(ratio),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.imshow("inputFrame",frame)
        cv2.imshow("outputImage",outputImage)
        cv2.imwrite(outputPath+str(counter)+".png",frame)
        if 27 == cv2.waitKey(1):
            break

        

    else: 
        print("skip",counter)
        break
cap.release()



