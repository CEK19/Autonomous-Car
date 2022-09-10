import os
from sys import prefix

imgPath = "C:/Users/Admin/Documents/coding/Autonomous-Car/dataset/dataset_kitty/data_road/training/image_2/"
calibPath = "C:/Users/Admin/Documents/coding/Autonomous-Car/dataset/dataset_kitty/data_road/training/calib/"
outputPath = "C:/Users/Admin/Documents/coding/Autonomous-Car/dataset/dataset_kitty/data_road/results/BIV"
prefix = "uu_0000"
posfix = ".png"

def addPath(stringPath):
    return "'" + stringPath + "'"

for i in range(0, 98):
    if(i<10):
        img_addr = imgPath + prefix + "0" + str(i) + posfix
    else:
        img_addr = imgPath + prefix + str(i) + posfix
    # print(img_addr)
    cmd = "python transform2BEV.py " + img_addr + " " + calibPath + " " + outputPath
    cmd = str(cmd)
    print("\n\n")
    print(cmd)
    os.system(cmd)