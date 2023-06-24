import os
import shutil
import cv2

for each in os.listdir("D:/container/AI_DCLV/readData/labeled_v7/predic"):
    frame = cv2.imread("D:/container/AI_DCLV/readData/labeled_v7/output/"+each)
    cv2.imwrite("D:/container/AI_DCLV/readData/labeled_v7/label/"+each,frame)

    frame = cv2.imread("D:/container/AI_DCLV/readData/labeled_v7/image_org/"+each)
    cv2.imwrite("D:/container/AI_DCLV/readData/labeled_v7/image/"+each,frame)