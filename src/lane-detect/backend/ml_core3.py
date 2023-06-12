import cv2
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt
import re
import sys


class lineEdge:
    def __init__(self, pointList):
    # y = ax + b
        self.a = (pointList[3]-pointList[2])/(pointList[1]-pointList[0])
        self.b = (pointList[2]-self.a*pointList[0])
        
dataSet = "subFull"

imgPath = "D:/container/AI_DCLV/readData/imageSet/object/image/"
listImg = os.listdir(imgPath)

listImg.sort(key = len)

for index in range(len(listImg)):
    name = listImg[index]
    frame = cv2.imread(imgPath+name)
    frame = cv2.resize(frame,(128,128))
    # print(name)
    label = cv2.imread("D:/container/AI_DCLV/readData/imageSet/object/predic/"+name)
    label = cv2.resize(label,(128,128))
    timeStart = time.time()
    label = cv2.cvtColor(label,cv2.COLOR_RGB2GRAY)
    # cv2.imwrite("outputDemo/label-"+name,label)
    # label = cv2.adaptiveThreshold(label, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    # cv2.imwrite("outputDemo/thres-"+name,label)
    lines = cv2.HoughLines(label, 1, np.pi / 180, 30, None, 0, 0)
    if lines is None:
        continue
    print(len(lines))

    leftLaneData = []
    rightLaneData = []

    

    ratio = 0
        
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
            # t = (128-rho*b)/a
            
            if lines[i][0][1] < 1:
                # cv2.line(frame, pt1, pt2, (50,0,255), 1, cv2.LINE_AA)
                # leftLaneData.append([x0 + t*(-b),lines[i][0][1]])
                leftLaneData.append(lines[i][0])
            elif lines[i][0][1] < 2:
                # cv2.line(frame, pt1, pt2, (0,100,0), 1, cv2.LINE_AA)
                continue
            else:
                # cv2.line(frame, pt1, pt2, (255,0,50), 1, cv2.LINE_AA)
                # rightLaneData.append([x0 + t*(-b),lines[i][0][1]])
                rightLaneData.append(lines[i][0])
    # cv2.imwrite("outputDemo/hough-"+name,frame)
    
    
    # blankImage = np.zeros((128,256,3),dtype="uint8")
    label = cv2.cvtColor(label,cv2.COLOR_GRAY2RGB)
    # frame = cv2.hconcat([blankImage,frame,label])

    if len(leftLaneData) != 0 and len(rightLaneData) != 0:

        leftLaneData = np.array(leftLaneData)
        leftLaneData = [np.mean(leftLaneData[:,0]),np.mean(leftLaneData[:,1]),np.std(leftLaneData[:,0]),np.std(leftLaneData[:,1])]

        rightLaneData = np.array(rightLaneData)
        rightLaneData = [np.mean(rightLaneData[:,0]),np.mean(rightLaneData[:,1]),np.std(rightLaneData[:,0]),np.std(rightLaneData[:,1])]
        # print(leftLaneData)



        for eachLane in [leftLaneData,rightLaneData]:
            # print([leftLaneData,rightLaneData]," --- ",eachLane)
            header = "right Lane: "
            y1 = 60
            if eachLane == leftLaneData:
                header = "left Lane: "
                y1 = 90
            # frame = cv2.putText(frame, header+str(eachLane[2])[:6] + " / "+ str(eachLane[3])[:6], (5,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
            rho = eachLane[0]
            theta = eachLane[1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            x = (128 - y0)/a
            x = x0 -b*x
            if theta < 1:
                cv2.line(frame, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
            elif theta < 2:
                # cv2.line(frame, pt1, pt2, (0,255,255), 2, cv2.LINE_AA)
                pass
            else:
                cv2.line(frame, pt1, pt2, (255,0,0), 2, cv2.LINE_AA)

    # cv2.imshow("frame",frame)
    print(listImg[index])
    cv2.imwrite("D:/container/AI_DCLV/readData/imageSet/object/backend-t/"+listImg[index],frame)

    # cv2.imwrite("outputDemo/"+name,frame)
    # print(name)
    # if ratio <= 0:
    #     ratio = -1
    processTime = time.time() - timeStart
    # f.writelines(str(processTime))

    # if cv2.waitKey(100) == 27:
    #     break
    # cv2.waitKey(0)
    # break