import random
import cv2
import numpy as np
import os

pathOutput = "D:/container/AI_DCLV/readData/compare_old/"

def mousePoints(event,x,y,flags,params):
      global counter
      # Left button click
      if event == cv2.EVENT_LBUTTONDOWN:
        print("mouse",x,y)
        polyPoint.append([x,y])
      

frameList = []
nameList = os.listdir("D:/container/AI_DCLV/readData/compare_new/image")

# nameList = ["1520.png"]

for name in nameList:
  frame = cv2.imread("D:/container/AI_DCLV/readData/compare_new/image/"+name)
  frameList.append(frame)
  print(name)

polyPoint = []

print("GO")

while True:
  valid = False
  while valid != True:
    counter = int(random.random()*len(nameList))
    print(counter,nameList[counter])
    # tmp = cv2.imread()
    valid = not os.path.isfile(pathOutput+"label/"+nameList[counter])

  frame = cv2.resize(frameList[counter],(512,512))
  cv2.imshow('Frame',frame)
  cv2.setMouseCallback("Frame", mousePoints)

  # Display the resulting frame
  frame = cv2.resize(frame,(512,512))
  blankImg = np.zeros((212,512,3),dtype = "uint8")
  frame2 = cv2.vconcat([frame,blankImg])
  cv2.imshow('Frame',frame2)
  frame = frame[0:512,:]

  print(counter)
  if cv2.waitKey(0) == 27:
    break
  
  label = np.zeros((512,512))
  polyPoint = np.array(polyPoint)
  cv2.fillPoly(label, pts = [polyPoint], color =(255,255,255))
  cv2.imshow("filledPolygon", label)
  
  cv2.imwrite(pathOutput+"image/"+nameList[counter],frame)
  cv2.imwrite(pathOutput+"label/"+nameList[counter],label)
  polyPoint = []
  # Break the loop
  # break

# Closes all the frames
cv2.destroyAllWindows()