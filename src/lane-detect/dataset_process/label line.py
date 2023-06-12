import random
import cv2
import numpy as np
import os
# not "/" at the end
pathOutput = "D:/container/AI_DCLV/readData/labeled_v8"
dataSource = "D:/container/AI_DCLV/readData/labeled_v8/image"

index = 0

def mousePoints(event,x,y,flags,params):
      global pre
      global frame
      global label
      # Left button click
      if event == cv2.EVENT_LBUTTONDOWN:
        print("mouse",x,y)
        if len(pre) == 0:
          pre = [x,y]
        else:
           frame = cv2.line(frame,pre,[x,y],(0,0,255),5,cv2.LINE_AA)
           label = cv2.line(label,pre,[x,y],255,5,cv2.LINE_AA)
           pre = []
           cv2.imshow("Frame",frame)
      
pre = []
frameList = []
nameList = os.listdir(dataSource)

polyPoint = []

print("GO")

while True:
  valid = False
  while valid != True:
    counter = int(random.random()*len(nameList))
    print(counter,nameList[counter])
    valid = not os.path.isfile(pathOutput+"/label/"+nameList[counter])
  label = np.zeros((512,512),dtype='uint8')

  frame = cv2.resize(cv2.imread(dataSource+"/"+nameList[counter]),(512,512))
  orgFrame = frame.copy()
  index += 1
  print("Good job bro!, u can do it! You have done: ", index, "pic")
  cv2.imshow('Frame', frame)
  cv2.setMouseCallback("Frame", mousePoints)

  # Display the resulting frame
  frame = cv2.resize(frame,(512,512))
  cv2.imshow('Frame',frame)

  print(counter)
  if cv2.waitKey(0) == 27:
    break
  
  
  # cv2.imshow("filledPolygon", label)
  
  cv2.imwrite(pathOutput+"/image/"+nameList[counter],orgFrame)
  cv2.imwrite(pathOutput+"/label/"+nameList[counter],label)
  polyPoint = []
  # Break the loop
  # break

# Closes all the frames
cv2.destroyAllWindows()