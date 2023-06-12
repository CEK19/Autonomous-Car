import cv2
import numpy as np

from os import listdir
from os.path import isfile, join

# for i in range(1600):
#     cv2.imwrite("/Users/lap15864-local/temp/Autonomous-Car/report/tu/cnn_black/f" + str(i) + ".png", np.zeros((300,300)))

names = []

myPath = "/Users/lap15864-local/temp/Autonomous-Car/report/tu/cnn"
onlyfiles = [f for f in listdir(myPath) if isfile(join(myPath, f))]
for fileName in onlyfiles:
	if fileName[0] == ".":
		continue
	fileName = fileName[1:]
	fileName = fileName[:-4]
	names.append(int(fileName))

names.sort()

for i in range(len(names)):
    if i == 0:
        continue
    
    if names[i] != names[i-1] + 1:
        print(names[i-1], names[i])
        break