import numpy as np
import cv2

video = cv2.VideoCapture('./vid1.avi')
_, firstFrame = video.read()
while(1):
    _, frame = video.read()    
    frame = cv2.absdiff(firstFrame, frame)
    firstFrame = frame
    cv2.imshow("window", frame)
    key = cv2.waitKey(50)
    if (key == 27):
        break
video.release()
cv2.destroyAllWindows()