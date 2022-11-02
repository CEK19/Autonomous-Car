import numpy as np
import cv2
import time
import os

def preprocessing(frame):    
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # sensitivity = 120
    sensitivity = 150
    lowerWhite = np.array([0,0,255-sensitivity])
    upperWhite = np.array([255,sensitivity,255])
    maskWhite = cv2.inRange(hsvImg, lowerWhite, upperWhite)
    cv2.imshow("white filter", maskWhite)
    
    # lowerYellow = np.array([25,100 ,20]) #H,S,V
    # upperYellow = np.array([32, 255,255]) #H,S,V
    
    lowerYellow = np.array([12,100 ,20]) #H,S,V
    upperYellow = np.array([23, 255,255]) #H,S,V    
    maskYellow = cv2.inRange(hsvImg, lowerYellow, upperYellow)    
    cv2.imshow("yellow filter", maskYellow)
    
    combineColorImg = cv2.bitwise_or(maskWhite, maskYellow)
    cv2.imshow("Combine color filter", combineColorImg)

    copyOriginalFrame = frame.copy()
    copyOriginalFrame[np.where(combineColorImg==[0])] = [0]
    
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)    
    cannyEdgeDectection = cv2.Canny(blurred, 100, 180)
    cv2.imshow("canny edge", cannyEdgeDectection)
    
    combineFirst = cv2.bitwise_or(cannyEdgeDectection, combineColorImg)
    
    backtorgb = cv2.cvtColor(combineFirst ,cv2.COLOR_GRAY2RGB)    
    return cv2.vconcat([backtorgb, frame])

def videoReading ():
    # VIDEO READING
    cap = cv2.VideoCapture("/Users/mac/Desktop/vid4nhan.avi")
    while(cap.isOpened()):
        begin = time.time()
        _, frame = cap.read()     
        begin = time.time()
        result = preprocessing(frame)        
        cv2.imshow("chac la ko gion dau", result)
    
        end = time.time()
        print(1/(end-begin))                
        key = cv2.waitKey(10)
        if (key == 32):
            cv2.waitKey()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    
def imageReading():
    # VIDEO READING
    PATH_IMAGE_INPUT = "/Users/mac/Desktop/drive-download-20221007T154316Z-001/"
    imageCollections = os.listdir(PATH_IMAGE_INPUT)
    PATH_IMAGE_OUTPUT = "/Users/mac/Desktop/output1/" 
    
    for image in imageCollections:
        begin = time.time()
        frame = cv2.imread(PATH_IMAGE_INPUT + image, cv2.IMREAD_COLOR)        
        #------------------------- PREPROCESSING IMAGE ----------------------------
        resultPreprocessing = preprocessing(frame)
        cv2.imwrite(PATH_IMAGE_OUTPUT + image, resultPreprocessing)
        end = time.time()
        print(1/(end-begin))
        
imageReading()
# videoReading()
# cv2.destroyAllWindows()