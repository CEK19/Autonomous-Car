import math
import numpy as np
import cv2 as cv
import keyboard

GRAD_MIN_THRESH_HOLD = 20
GRAD_MAX_THRESH_HOLD = 180
S_THRESH_HOLD = (80, 150)
V_THRESH_HOLD = (80, 150)


def abs_sobel_thresh(rgbImg, orient='x', threshMin=25, threshMax=255):
    # convert RGB to HLS
    # ROW_IMG = rgbImg.rows
    # COL_IMG = rgbImg.cols
    ROWS, COLS = rgbImg.shape

    print("rows: ", ROWS)
    print("cols: ", COLS)


def preprocessImg(orgImg):
    img = cv.cvtColor(orgImg, cv.COLOR_BGR2GRAY)
    ret, threshImg = cv.threshold(img, 127, 255, 0)
    abs_sobel_thresh(img)

    return threshImg

def thicken(img, orgImg, thickness=2):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (255, 255, 255), thickness=thickness)
    cv.drawContours(orgImg, contours, -1, (255, 255, 255), thickness=thickness)
    
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (255, 255, 255), thickness=cv.FILLED)
    cv.drawContours(orgImg, contours, -1, (255, 255, 255), thickness=cv.FILLED)

def contourFunction(threshImg, orgImg):
    # contours, hierarchy = cv.findContours(threshImg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours, hierarchy = cv.findContours(threshImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    

    contoursList = list(contours)
    largeGr = []
    smallGr = []

    index = 0
    while (index < len(contoursList)):
        if(cv.contourArea(contours[index]) < 1.2):
            smallGr.append(contoursList[index])
        else:
            largeGr.append(contoursList[index])
            # contoursList.pop(index)
        index += 1

    cv.drawContours(orgImg, smallGr, -1, (0, 0, 0), thickness=cv.FILLED)
    cv.drawContours(threshImg, smallGr, -1, (0, 0, 0), thickness=cv.FILLED)
   
showIndex = 0;
 
def contourFunction2 (img, orgImg):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largeGr = list(contours)
    squareGr = []
    i = 0
    while (i < len(largeGr)):
        cnt = largeGr[i]
        x,y,w,h = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        dense = float( area ) / (w*h)
        
        i += 1
        # if float(w)/h < 1.5 and float(w)/h > 0.75:
        # if 0.75 < float(w)/h < 1.1 and w > 3:
        #     squareGr.append(cnt)
        
        # if dense > 0.2 and w > 4 and h < 20:
        #     squareGr.append(cnt)
        
        if dense > 0.5 and ( (w > 8 and area < 1000) or (0.45 < area < 1000 and 8 < w and h < 10 ) ) :
            print("i: ", i,", Area: ", float( cv.contourArea(cnt) ), " / Box: ", (w*h), " = ", dense)
            squareGr.append(cnt)
    
    # global showIndex
    showIndex = 7
    print("----> ", "area: ", cv.contourArea(squareGr[showIndex]), " dimention: ", cv.boundingRect(squareGr[showIndex]), " dense: ", float(area)/(w*h))
    cv.drawContours(orgImg, squareGr, showIndex, (255, 0, 0), thickness=cv.FILLED)
    cv.drawContours(threshImg, squareGr, showIndex, (0, 0, 0), thickness=cv.FILLED)


####################
##      MAIN      ##
####################
index = 0

while True:
    if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
        print("quiting Tu function . . .")
        break
    elif keyboard.is_pressed('left') or keyboard.is_pressed('down'):
        index -= 1
        showIndex -= 1
        if index == -1:
            break
    else:
        cv.destroyAllWindows()
        showIndex += 1
        index += 1
        print(showIndex)
        if index == 19:
            break

    # orgImg = cv.imread(str(index) + '.jpg')
    orgImg = cv.imread('4.jpg')
    # orgImg = cv.imread("../data_road/training/image_2/um_000000.png")
    threshImg = preprocessImg(orgImg)
    cv.imshow("before", threshImg)

    contourFunction(threshImg, orgImg)
    cv.imshow("contour1", orgImg)
    thicken(threshImg, orgImg, 2)
    cv.imshow("thicken", orgImg)
    contourFunction2(threshImg, orgImg)
    cv.imshow("contour2", orgImg)

    cv.waitKey(0)
