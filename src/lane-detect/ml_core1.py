import cv2
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt

class lineEdge:
    def __init__(self, pointList):
    # y = ax + b
        self.a = (pointList[3]-pointList[2])/(pointList[1]-pointList[0])
        self.b = (pointList[2]-self.a*pointList[0])
        
dataSet = "st2"

imgPath = "D:/container/AI_DCLV/readData/output/"+dataSet+"/img/"
listImg = os.listdir(imgPath)

counter = 0
avgTime = 0

for name in listImg:
    frame = cv2.imread(imgPath+name)
    print(name)
    label = cv2.imread("D:/container/AI_DCLV/readData/output/"+dataSet+"/label/"+name)

    timeStart = time.time()
    label = cv2.cvtColor(label,cv2.COLOR_RGB2GRAY)
    label = cv2.adaptiveThreshold(label, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    lines = cv2.HoughLines(label, 1, np.pi / 180, 150, None, 0, 0)

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
                cv2.line(frame, pt1, pt2, (50,0,255), 1, cv2.LINE_AA)
                # leftLaneData.append([x0 + t*(-b),lines[i][0][1]])
                leftLaneData.append(lines[i][0])
            elif lines[i][0][1] < 2:
                cv2.line(frame, pt1, pt2, (0,100,0), 1, cv2.LINE_AA)
                continue
            else:
                cv2.line(frame, pt1, pt2, (255,0,50), 1, cv2.LINE_AA)
                # rightLaneData.append([x0 + t*(-b),lines[i][0][1]])
                rightLaneData.append(lines[i][0])


    processTime = time.time() - timeStart
    avgTime = avgTime*0.9 + processTime*0.1*1000
    
    blankImage = np.zeros((512,512,3),dtype="uint8")
    frame = cv2.hconcat([blankImage,frame,blankImage])

    if len(leftLaneData) != 0 and len(rightLaneData) != 0:

        leftLaneData = np.array(leftLaneData)
        leftLaneData = [np.mean(leftLaneData[:,0]),np.mean(leftLaneData[:,1])]

        rightLaneData = np.array(rightLaneData)
        rightLaneData = [np.mean(rightLaneData[:,0]),np.mean(rightLaneData[:,1])]
        # print(leftLaneData)



        for eachLane in [leftLaneData,rightLaneData]:
            # print([leftLaneData,rightLaneData]," --- ",eachLane)
            rho = eachLane[0]
            theta = eachLane[1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)+512), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)+512), int(y0 - 1000*(a)))
            
            if theta < 1:
                cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            elif theta < 2:
                cv2.line(frame, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)
            else:
                cv2.line(frame, pt1, pt2, (255,0,0), 3, cv2.LINE_AA)

    frame = cv2.putText(frame, "t: "+str(avgTime)[:6]+" ms", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow("frame",frame)
    cv2.imshow("label",label)

    if cv2.waitKey(25) == 27:
        break
    # cv2.waitKey(0)
    # break