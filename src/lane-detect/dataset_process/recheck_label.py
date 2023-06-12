import random
import cv2
import numpy as np
import os
# not "/" at the end
pathOutput = "D:/container/AI_DCLV/readData/labeled_v10"
pathInput = "D:/container/AI_DCLV/readData/labeled_v8"

index = 0

def mousePoints(event,x,y,flags,params):
      global pre
      global frame
      global label
      # Left button click
      if event == cv2.EVENT_LBUTTONDOWN:
        # print("mouse",x,y)
        if len(pre) == 0:
          pre = [x,y]
        else:
           frame = cv2.line(frame,pre,[x,y],(0,0,255),5,cv2.LINE_AA)
           label = cv2.line(label,pre,[x,y],255,5,cv2.LINE_AA)
           pre = []
           cv2.imshow("Frame",frame)
      
pre = []
frameList = []
nameList = os.listdir(pathInput+"/orgimg")
tmp = []
for each in nameList:
   if "ras5" in each:
      tmp.append(each)
nameList = tmp
print(tmp)

polyPoint = []

print("GO")

while True:
  valid = False
  while valid != True:
    counter = int(random.random()*len(nameList))
    print(nameList[counter])
    valid = not os.path.isfile(pathOutput+"/label/"+nameList[counter])
  label = cv2.resize(cv2.imread(pathInput+"/output/"+nameList[counter]),(512,512))
  label = cv2.cvtColor(label,cv2.COLOR_RGB2GRAY)

  frame = cv2.resize(cv2.imread(pathInput+"/predic/"+nameList[counter]),(512,512))
  orgFrame = cv2.resize(cv2.imread(pathInput+"/orgimg/"+nameList[counter]),(512,512))
  index += 1
  
  # Display the resulting frame
  frame = cv2.resize(frame,(512,512))
  cv2.imshow('Frame',frame)
  cv2.setMouseCallback("Frame", mousePoints)

  key  = cv2.waitKey(0)
  if key == 27:
    break
  elif key == 114:
    label = np.zeros((512,512),dtype='uint8')

    frame = cv2.resize(cv2.imread(pathInput+"/orgimg/"+nameList[counter]),(512,512))
    orgFrame = cv2.resize(cv2.imread(pathInput+"/orgimg/"+nameList[counter]),(512,512))
    cv2.imshow('Frame',frame)
    key  = cv2.waitKey(0)
    if key == 27:
      break
  # cv2.imshow("filledPolygon", label)
  
  cv2.imwrite(pathOutput+"/image/"+nameList[counter],orgFrame)
  cv2.imwrite(pathOutput+"/label/"+nameList[counter],label)
  polyPoint = []
  # Break the loop
  # break

# Closes all the frames
cv2.destroyAllWindows()