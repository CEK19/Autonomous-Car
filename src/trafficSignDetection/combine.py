import cv2
from os import listdir
from os.path import isfile, join
import numpy as np

yoloPath = "/Users/lap15864-local/temp/Autonomous-Car/report/tu/yolo"
cnnPath = "/Users/lap15864-local/temp/Autonomous-Car/report/tu/cnn"

# yolo wh
height = 240
width = 320

# cnn wh
signHeight = 80
signWidth = 80




yoloFiles = [f for f in listdir(yoloPath) if isfile(join(yoloPath, f))]
cnnFiles = [f for f in listdir(cnnPath) if isfile(join(cnnPath, f))]

print(len(yoloFiles), len(cnnFiles))


for i in range(len(yoloFiles)):
    img = cv2.imread(yoloPath + "/f" + str(i+1) + ".png")
    sign = cv2.imread(cnnPath + "/f" + str(i+1) + ".png")
    print(img.shape, sign.shape)
    print(i)
    if (np.sum(sign) == 0):
        cv2.imwrite("/Users/lap15864-local/temp/Autonomous-Car/report/tu/combine/f" + str(i+1) + ".png", img)
    else:
        sign = cv2.resize(sign, (signHeight, signWidth))
        img[0:signHeight+6, 0:signWidth+6] = [0,0,0]
        img[3:signHeight+3, 3:signWidth+3] = sign
        cv2.imwrite("/Users/lap15864-local/temp/Autonomous-Car/report/tu/combine/f" + str(i+1) + ".png", img)




	