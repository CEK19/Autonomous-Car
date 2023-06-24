from multiprocessing.connection import wait
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt 
import os

for vidCounter in range(11,17):
    dataPath = "D:/NDT/PY_ws/DCLV/src/dataset/video/"+str(vidCounter)+".mp4"

    # nameList = os.listdir(dataPath+"image/")

    cap = cv2.VideoCapture(dataPath)

    ret = True

    counter = len(os.listdir("./output"))

    while ret:
        for _ in range(10):
            if not ret:
                break
            ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame,(128,128))
        cv2.imwrite("./output/ras3-"+str(counter)+".png",frame)
        counter += 1
        print(counter)






