import numpy as np
import cv2
import random
import matplotlib.pyplot as plt 
import os

counter = 1700
cap = cv2.VideoCapture("D:/Download/output2.avi")
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

while(cap.isOpened()):
    # for _ in range(3):
    ret, frame = cap.read()
    if ret == True:
        counter += 1
        print(counter)
        cv2.imwrite('D:/NDT/PY_ws/DCLV/ROS_zone/newBackend_1/data2/test2-'+str(counter)+".png",frame)
    else:
        break

cap.release()

cv2.waitKey(0)

