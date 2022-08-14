import math
import numpy as np
import cv2 as cv
import keyboard

GRAD_MIN_THRESH_HOLD = 20
GRAD_MAX_THRESH_HOLD = 180
S_THRESH_HOLD = (80, 150)
V_THRESH_HOLD = (80, 150)

X_MAX = 767
Y_MAX = 767

BLACK_COLOR = 0
WHITE_COLOR = 255


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

def isOutOfRange (point, type):
    if type == "x-axios":
        if point < 0 or point > X_MAX:
            return True
        else:
            return False
    elif type == "y-axios":
        if point < 0 or point > Y_MAX:
            return True
        else:
            return False
    elif type == "both":
        condition1 = point.x < 0 or point.x > X_MAX
        condition2 = point.y < 0 or point.y > Y_MAX
        if condition1 or condition2:
            return True
        else:
            return False
    else:
        print("Your type is invalid, please choose among 'x-axios', 'y-axios', 'both'")
    
def distance(xA, yA, xB, yB):
    return math.sqrt( (xA-xB)**2 + (yA-yB)**2 )

def density(threshImg, orgImg, centerPointX, centerPointY, radius):
    totalPoint = 0
    countedPoint = 0
    # tempImg = orgImg.copy()
    for y in range (centerPointY - radius, centerPointY + radius, 1):
        if isOutOfRange(y, "y-axios"):
            continue
        for x in range (centerPointX - radius, centerPointX + radius, 1):
            if isOutOfRange(x, "x-axios"):
                continue
            if distance(x, y, centerPointX, centerPointY) > radius:
                continue
            if threshImg[y][x] == WHITE_COLOR:
                countedPoint += 1
            totalPoint += 1
            # tempImg[y, x] = [255, 0, 0]

    # print(countedPoint, totalPoint, float(countedPoint) / totalPoint)
    # cv.imshow("temp", tempImg)
    return float(countedPoint) / totalPoint

# first contour function
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

# do this after thicken
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
    showIndex = -1
    # print("----> ", "area: ", cv.contourArea(squareGr[showIndex]), " dimention: ", cv.boundingRect(squareGr[showIndex]), " dense: ", float(area)/(w*h))
    cv.drawContours(orgImg, squareGr, showIndex, (255, 0, 0), thickness=cv.FILLED)
    cv.drawContours(threshImg, squareGr, showIndex, (0, 0, 0), thickness=cv.FILLED)

def contourFunction3(img, orgImg):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    
    standAloneGr = []
    
    print(len(contours))
    
    # for cnt in contours:
    i = 0
    while i < len(contours):
        print(i)
        cnt = contours[i]
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if cy < 500:
            i += 1
            continue
        # radius = 10, tham so = 0.12 => khá ổn
        if(density(img, orgImg, cx, cy, 10) < 0.09):
            standAloneGr.append(cnt)
        i += 1
    
    cv.drawContours(threshImg, standAloneGr, -1, (0, 0, 0), thickness=cv.FILLED)
    cv.drawContours(orgImg, standAloneGr, -1, (255, 0, 0), thickness=cv.FILLED)
    # for cnt in contours:
    #     x,y,w,h = cv.boundingRect(cnt)
        
    
    return 0


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

    orgImg = cv.imread(str(index) + '.jpg')
    # orgImg = cv.imread('3.jpg')
    threshImg = preprocessImg(orgImg)
    cv.imshow("before", threshImg)

    contourFunction(threshImg, orgImg)
    cv.imshow("contour1", orgImg)
    
    contourFunction3(threshImg, orgImg)
    cv.imshow("contour3", orgImg)
    
    
    # orgImg[100:150, 400:450] = [255, 0, 0]
    # cv.imshow("contours", orgImg)
    
    
    # thicken(threshImg, orgImg, 2)
    # cv.imshow("thicken", orgImg)
    
    # contourFunction2(threshImg, orgImg)
    # cv.imshow("contour2", orgImg)

    cv.waitKey(0)
