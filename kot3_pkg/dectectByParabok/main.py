import cv2 as cv
from matplotlib.ft2font import HORIZONTAL
import numpy as np
from matplotlib import pyplot as plt
import sys
import numpy

# def lanesDetection(img):
#     # img = cv.imread("./img/road.jpg")
#     # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#     # print(img.shape)
#     height = img.shape[0]
#     width = img.shape[1]

#     region_of_interest_vertices = [
#         (200, height), (width/2, height/1.37), (width-300, height)
#     ]
#     gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#     edge = cv.Canny(gray_img, 50, 100, apertureSize=3)
#     cropped_image = region_of_interest(
#         edge, np.array([region_of_interest_vertices], np.int32))

#     lines = cv.HoughLinesP(cropped_image, rho=2, theta=np.pi/180,
#                            threshold=50, lines=np.array([]), minLineLength=10, maxLineGap=30)
#     image_with_lines = draw_lines(img, lines)
#     # plt.imshow(image_with_lines)
#     # plt.show()
#     return image_with_lines

WHITE_COLOR = 255
BLACK_COLOR = 0


def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img = cv.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def ROIProcess(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = (255)
    cv.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

def laneDetection(img):
    VERTICAL = img.shape[0]
    HORIZONTAL = img.shape[1]
    
    grayImage = cv.cvtColor(img, cv.COLOR_RGB2GRAY)    
    cv.imshow("Gray" , grayImage)        
    
    # https://stackoverflow.com/questions/21324950/how-can-i-select-the-best-set-of-parameters-in-the-canny-edge-detection-algorith
    highThresh, thresh_im = cv.threshold(grayImage, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lowThresh = 0.5*highThresh    
    edgeDetectionImage = cv.Canny(grayImage, lowThresh, highThresh, 
                                  apertureSize=3)        
    cv.imshow("egde", edgeDetectionImage)
    lines = cv.HoughLinesP(edgeDetectionImage, 
                           rho=2, theta=np.pi/180,
                           threshold=30, lines=np.array([]), 
                           minLineLength=10, maxLineGap=50)
    imageWithLine = draw_lines(img, lines)    
    cv.imshow("Line hough transfom", imageWithLine)
    
    
    statisticMatrix = np.sum(edgeDetectionImage, axis=0) # CACULATE ALL COLUMN IN MATRIX
    leftStartPointX = np.argmax(statisticMatrix[0 : int(HORIZONTAL/2)])
    leftStartPointY = 0
    for index in range(VERTICAL-1 , 0 , -1):
        if (edgeDetectionImage[index][leftStartPointX] != BLACK_COLOR):
            leftStartPointY = index
            break
        
    rightStartPointX = int(HORIZONTAL/2) - 1 + np.argmax(statisticMatrix[int(HORIZONTAL/2 + 1) : int(HORIZONTAL)])    
    rightStartPointY = 0
    for index in range(VERTICAL-1 , 0 , -1):
        if (edgeDetectionImage[index][rightStartPointX] != BLACK_COLOR):
            rightStartPointY = index
            break    
    
    print((leftStartPointX, leftStartPointY))
    print((rightStartPointX, rightStartPointY))
    
    imageCircle = cv.circle(img, (leftStartPointX, leftStartPointY), 
                            20, (0, 255, 0), 10)
    imageCircle = cv.circle(imageCircle, (rightStartPointX, rightStartPointY), 
                            20, (0, 255, 0), 10)
    imageCircle = cv.circle(imageCircle, (0, 0), 
                        20, (0, 255, 0), 10)
    cv.imshow("Circle for fun", imageCircle) 
    cv.waitKey(0)
    cv.destroyAllWindows() 
    
    
def videoLanes():
    cap = cv.VideoCapture('./img/Lane.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = laneDetection(frame)
        cv.imshow('Lanes Detection', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def imageLanes():
    img = cv.imread("./img/demo/page3.png")
    processedImg = laneDetection(img=img)


if __name__ == "__main__":
    # videoLanes()
    imageLanes()
