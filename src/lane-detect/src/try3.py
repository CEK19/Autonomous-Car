from multiprocessing.connection import wait
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt 
import os

dataPath = "E:/DatasetContainer/AI_container/realData/dataset/cacheDataset/"
odataPath = "E:/DatasetContainer/AI_container/realData/dataset/"

nameList = os.listdir(dataPath+"image/")

for each in nameList:
    img = cv2.imread(dataPath+"image/"+each)
    lab = cv2.imread(dataPath+"label/"+each)
    # cv2.imshow(each,cv2.addWeighted(lab,0.5,img,1,0))
    cv2.imwrite(odataPath+"image/thinh3-"+each,img)
    cv2.imwrite(odataPath+"label/thinh3-"+each,lab)
    # cv2.waitKey(100)
    print(each)

cv2.waitKey(0)




