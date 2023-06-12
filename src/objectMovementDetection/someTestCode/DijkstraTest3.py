import cv2
import numpy as np

def genData(size):
    frame = np.zeros((size,size),dtype='uint8')
    circleRadius = 25
    offset = np.random.randint(1,25)
    circleCenter = (size//2+offset,np.random.randint(size//2-150,size//2+150))

    offset = circleRadius - offset

    frame = cv2.circle(frame,circleCenter,circleRadius,100,-1)
    # up line
    cv2.line(frame,(size//2,0),(size//2,circleCenter[1]-circleRadius-offset*3),200,1)
    cv2.line(frame,(size//2,circleCenter[1]-circleRadius-offset*3),(size//2-offset,circleCenter[1]-circleRadius),200,1)
    cv2.line(frame,(size//2-offset,circleCenter[1]-circleRadius),(size//2-offset,circleCenter[1]+circleRadius),200,1)
    cv2.line(frame,(size//2,circleCenter[1]+circleRadius+offset*3),(size//2-offset,circleCenter[1]+circleRadius),200,1)
    cv2.line(frame,(size//2,size),(size//2,circleCenter[1]+circleRadius+offset*3),200,1)

    # circleCenter = np.array(circleCenter)

    if np.random.randint(2) == 0:
        frame = cv2.flip(frame,1)
        circleCenter = [size - circleCenter[0],circleCenter[1]]

    return circleCenter, frame



# print(graph["10"])
        