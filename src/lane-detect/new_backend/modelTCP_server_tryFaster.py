import socket
import cv2
import numpy as np
import json
import shutil
import time

# from LaneDetect_model_visual import *
from LaneDetect_model_v2 import *

def warp(p):
    global transformMatrix
    px = (transformMatrix[0][0]*p[0] + transformMatrix[0][1]*p[1] + transformMatrix[0][2]) / ((transformMatrix[2][0]*p[0] + transformMatrix[2][1]*p[1] + transformMatrix[2][2]))
    py = (transformMatrix[1][0]*p[0] + transformMatrix[1][1]*p[1] + transformMatrix[1][2]) / ((transformMatrix[2][0]*p[0] + transformMatrix[2][1]*p[1] + transformMatrix[2][2]))
    p_after = (int(px), int(py))
    return p_after

def lines_intersect(p1, p2, p3, p4):
    """
    Determines if two lines intersect within a 50x50 rectangle
    """
    # Extract the x and y coordinates of the points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Check if the lines intersect using cross products
    v1 = (x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)
    v2 = (x4-x3)*(y2-y3) - (y4-y3)*(x2-x3)
    v3 = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    v4 = (x2-x1)*(y4-y1) - (y2-y1)*(x4-x1)

    # Check if the signs of the cross products are different
    if (v1 * v2 < 0) and (v3 * v4 < 0):
        # Check if the intersection point is within the 50x50 rectangle
        intersection_x = (v1*(x2-x1)*(y3-y1) - (x4-x3)*(y1-y3)*(x2-x1)) / (v1*(x2-x1) - (x4-x3)*(y2-y1))
        intersection_y = (y1-y3+(y2-y1)*intersection_x/(x2-x1))
        if (0 <= intersection_x <= 50) and (0 <= intersection_y <= 50):
            return True

    return False

def errorCheck(a,b,c,d):

    # Calculate the slopes of the two line segments
    slope1 = (b[1] - a[1]) / (b[0] - a[0]+0.0001)
    slope2 = (d[1] - c[1]) / (d[0] - c[0]+0.0001)

    # Calculate the y-intercepts of the two line segments
    y_intercept1 = a[1] - slope1 * a[0]
    y_intercept2 = c[1] - slope2 * c[0]

    # Calculate the x-coordinate of the collision point
    x = (y_intercept2 - y_intercept1) / (slope1 - slope2)

    # Calculate the y-coordinate of the collision point
    y = slope1 * x + y_intercept1

    center1 = [(a[0]+b[0])/2,(a[1]+b[1])/2]
    center2 = [(c[0]+d[0])/2,(c[1]+d[1])/2]

    # Return the collision point as a list of two floats
    if (0<x<50 and 0<y<50):
        return False, math.sqrt(((center1[0]-center2[0])**2+(center1[1]-center2[1])**2))
    else:
        return True, math.sqrt(((center1[0]-center2[0])**2+(center1[1]-center2[1])**2))
    
def extenLine(pA, pB):
    
    a, b = np.polyfit([pA[0],pB[0]], [pA[1],pB[1]], 1)        
    x1 = 0
    y1 = int(a*x1 + b)
    x2 = 50 - 1
    y2 = int(a*x2 + b)
    return (x1,y1),(x2,y2)

def line_equation(x0, y0, x1, y1):
    # y = ax + b
    a = (y1 - y0) / (x1 - x0 + 0.000001)
    b = y0 - a * x0
    return [a, b]

def find_edge_intersections(width, height, a, b):
    """
    Find the intersection points of a line with the edges of an image
    Args:
        width (int): the width of the image
        height (int): the height of the image
        a (float): the slope of the line
        b (float): the y-intercept of the line
    Returns:
        A tuple of two arrays representing the (x, y) coordinates of the two intersection points
    """
    # Find the intersection points with the top and bottom edges of the image
    x1 = -b / a
    y1 = 0
    x2 = (height - b) / a
    y2 = height
    
    # Check if the line intersects with the left or right edges of the image
    if x1 < 0:
        # Find the intersection point with the left edge
        x1 = 0
        y1 = b
        
    elif x1 > width:
        # Find the intersection point with the right edge
        x1 = width
        y1 = a * x1 + b
        
    if x2 < 0:
        # Find the intersection point with the left edge
        x2 = 0
        y2 = b
        
    elif x2 > width:
        # Find the intersection point with the right edge
        x2 = width
        y2 = a * x2 + b
        
    # Create arrays of the intersection points
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    
    # Return the intersection points as a tuple
    return p1, p2

def getLineFormular(pt1,pt2):
    # ax+by=c
    #return a,b,c
    a = pt2[1] - pt1[1]
    b = pt1[0] - pt2[0]

    if a*b == 0:
        return False,0,0,0
    else:
        c = a*pt1[0] + b*pt1[1]
        return True, [int(c/a),0],[int((c - 50*b)/a),50],[a,b,c]
    
def find_collision_point(l1,l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    x = (b1*c2-c1*b2)/(a2*b1-a1*b2 + 0.00000001)
    y = (c1-a1*x)/(b1+0.00000001)
    return (x, y)

def keySortFunction(x):
    return x[1]

def closest_point_on_line(a, b, x0, y0):
    # y = ax + b, A(x0, y0) we want to find B(x1, y1) is the closest point on the line to A
    # calculate the x-coordinate of the closest point on the line
    x1 = (a*(y0-b) + x0) / (a**2 + 1)    
    # calculate the y-coordinate of the closest point on the line
    y1 = a*x1 + b    
    return int(x1), int(y1)

def errorCorrector(listPoint):
    w = 50
    h = 50
    A,B = listPoint[0]
    C,D = listPoint[1]
    valid1,A,B,l1 = getLineFormular(A,B)
    valid2,C,D,l2 = getLineFormular(C,D)

    if valid1 and valid2:
        colition = find_collision_point(l1,l2)
        if (0<colition[0]<50 and 0<colition[1]<50):
            return [[],[]]
        Left = [A,B]
        Right = [C,D]

        Left.sort(key=keySortFunction)
        Right.sort(key=keySortFunction)
        A,B = Left
        C,D = Right
        if (int(A[0]-B[0]) == 0):
            A[0] = A[0] + 0.0001
        if (int(C[0]-D[0]) == 0):
            C[0] = C[0] + 0.0001
        slopeAB, constAB = line_equation(A[0], A[1], B[0], B[1])        
        alphaAB = math.atan(slopeAB)
  
        slopeCD, constCD = line_equation(C[0], C[1], D[0], D[1])
        alphaCD = math.atan(slopeCD)
        
        isSlopeOppositeSign = slopeAB * slopeCD < 0
        
        newA, newB = find_edge_intersections(w, h, slopeAB, constAB)
        newC, newD = find_edge_intersections(w, h, slopeCD, constCD)        
        
        midPointAB = [(newA[0] + newB[0])/2, (newA[1] + newB[1])/2] # [x,y]
        midPointCD = [(newC[0]+newD[0])/2,(newC[1]+newD[1])/2] # [x,y]   
             
        
        avgSlope = math.tan((alphaAB + alphaCD) / 2 + (math.pi / 2 if isSlopeOppositeSign else 0))
        
        newSlopeAB = avgSlope
        newConstAB = midPointAB[1] - newSlopeAB * midPointAB[0]
        
        newSlopeCD = avgSlope
        newConstCD = midPointCD[1] - newSlopeCD * midPointCD[0]            
        
        topLeft, botLeft = find_edge_intersections(w, h, newSlopeAB, newConstAB) # [x,y]
        topRight, botRight = find_edge_intersections(w, h, newSlopeCD, newConstCD) # [x,y]

        pointFindDistanceA = list(map(int, midPointAB))
        pointFindDistanceB = closest_point_on_line(newSlopeCD, newConstCD, midPointAB[0], midPointAB[1])

        return [[topLeft,botLeft],[topRight,botRight],math.sqrt((pointFindDistanceA[0]-pointFindDistanceB[0])**2+(pointFindDistanceA[1]-pointFindDistanceB[1])**2)]
    else:
        return [[],[]]
    

def project_point_onto_line(xO, yO, xA, yA, xB, yB):
    # Compute the projection of point O onto the line passing through A and B
    dx = xB - xA
    dy = yB - yA
    dot_product = (xO - xA) * dx + (yO - yA) * dy
    projection = ((dot_product / (dx ** 2 + dy ** 2)) * dx, (dot_product / (dx ** 2 + dy ** 2)) * dy)
    # Compute the coordinates of the projected point H
    xH = xA + projection[0]
    yH = yA + projection[1]
    return xH, yH

def find_intersection_with_y_axis(xA, yA, xB, yB):
    m = (yB - yA) / (xB - xA + 0.00001)
    b = yA - m * xA
    x_intersection = -b / m
    return x_intersection

def perpendicular_points(A, B, k, fill_side='left'):
    xA, yA = A
    xB, yB = B
    # compute the distance from A to B
    AB = math.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)
    # compute the unit vector in the direction of AB
    u = ((xB - xA) / AB, (yB - yA) / AB)
    # compute the unit vector perpendicular to AB
    v = (-u[1], u[0])
    # compute the coordinates of points C
    xC1 = xA + k * v[0]
    yC1 = yA + k * v[1]
    xC2 = xA - k * v[0]
    yC2 = yA - k * v[1]
    
    xD1 = xB + k * v[0]
    yD1 = yB + k * v[1]
    xD2 = xB - k * v[0]
    yD2 = yB - k * v[1]    
    
    x_intersectionCD1 = find_intersection_with_y_axis(xC1, yC1, xD1, yD1)
    x_intersectionCD2 = find_intersection_with_y_axis(xC2, yC2, xD2, yD2)
    
    if (fill_side == "right"):
        return  (xC1, yC1, xD1, yD1) if x_intersectionCD1 > x_intersectionCD2 else (xC2, yC2, xD2, yD2)

    return (xC1, yC1, xD1, yD1) if x_intersectionCD1 < x_intersectionCD2 else (xC2, yC2, xD2, yD2)

def fixListPoint(listPoint,currentRoadLength):
    if (len(listPoint[0]) + len(listPoint[1]) == 2) and currentRoadLength  != -1 :
        if len(listPoint[0]) != 0:
            listPoint[0][0] = warp(listPoint[0][0])
            listPoint[0][1] = warp(listPoint[0][1])
            x1,y1,x2,y2 = perpendicular_points(listPoint[0][0], listPoint[0][1], currentRoadLength, "right")
            listPoint[1] = [[int(x1),int(y1)], [int(x2),int(y2)]]
        else:
            listPoint[1][0] = warp(listPoint[1][0])
            listPoint[1][1] = warp(listPoint[1][1])
            x1,y1,x2,y2 = perpendicular_points(listPoint[1][0], listPoint[1][1], currentRoadLength, "left")
            listPoint[0] = [[int(x1),int(y1)], [int(x2),int(y2)]]
    return listPoint

def drawTwoLane(frame, listPoint):
    frame = cv2.line(frame,listPoint[0][0],listPoint[0][1],(0,0,255),1,cv2.LINE_AA)
    frame = cv2.line(frame,listPoint[1][0],listPoint[1][1],(255,0,0),1,cv2.LINE_AA)
    return frame

HOST = '192.168.1.4'  
PORT = 8000        

shutil.rmtree("./his")
os.mkdir("./his")

shutil.rmtree("./his1")
os.mkdir("./his1")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(2)
print("Ready")
counter = 0
currentRoadLength = -1
client, addr = s.accept()
f = open("data.txt", "w")

srcPos = np.float32([(13,71),(113,76),(37,42),(91,43)])
dstPos = np.float32([(16,2),(38,2),(19,-21),(37,-21)])
transformMatrix = cv2.getPerspectiveTransform(srcPos, dstPos)

initTime = time.time()

while True:
    # try:
    print('Connected by', addr)
    frame = []
    cache = ""
    while True:
        data = client.recv(65536)
        # print("recv",len(data))
        # f.write(str(data)+"\n")
        time.sleep(0.001)
        if len(cache) == 0:
            cache = data
        else:
            cache = cache + data

        if len(cache) == 16384:
            frame = np.frombuffer(cache,'uint8')
            # frame2 = np.frombuffer(data2,'uint8')
            cache = ""
        else:
            continue
        print("Run frame ",counter)
        # print("recive time:",time.time()-initTime)
        frame = np.resize(frame,(128,128))
        # cv2.imshow("frame",frame)
        # if counter%2 == 0:
            # cv2.imwrite("./his1/ras4-"+str(counter)+".png",frame)
        predict,listPoint = runLaneDetectModel(frame,counter%30 == 0)
        returnDict = {}

        # tmp = cv2.resize(frame,(512,512))
        # cv2.imshow("-",tmp)

        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        visualMap = np.zeros((50,50,3),dtype='uint8')
        visualMap2 = visualMap.copy()
        
        if len(listPoint[0]) + len(listPoint[1]) == 4:
            '''
            if enought two lane, warp
            '''
            status = "full"
            frame = drawTwoLane(frame,listPoint)
            frame = cv2.resize(frame,(512,512))
            listPoint[0][0] = warp(listPoint[0][0])
            listPoint[0][1] = warp(listPoint[0][1])
            listPoint[1][0] = warp(listPoint[1][0])
            listPoint[1][1] = warp(listPoint[1][1])
        else:
            '''
            if only one lane, warp that lane and using currentRoadLength to generate the other lane
            '''
            status = "miss"
            fixListPoint(listPoint,currentRoadLength)
            
        if len(listPoint[0]) + len(listPoint[1]) == 4:
            '''
            if this one has four point, mean that the input has two lane, or one lane but it can generate the other
            '''
            visualMap = drawTwoLane(visualMap,listPoint)
            if status == "full":
                listPoint = errorCorrector(listPoint)
            returnDict = {}

        if len(listPoint[0]) + len(listPoint[1]) == 4:
            if currentRoadLength == -1:
                currentRoadLength = listPoint[2]
            elif status == "full":
                currentRoadLength = currentRoadLength*0.9 + listPoint[2]*0.1
            returnDict["tl"] = listPoint[0][0]
            returnDict["bl"] = listPoint[0][1]
            returnDict["tr"] = listPoint[1][0]
            returnDict["br"] = listPoint[1][1]

            visualMap2 = drawTwoLane(visualMap2,listPoint)

        # cv2.imshow("visual",visualMap)

        
            
        frame = cv2.resize(frame,(512,512))
        visualMap = cv2.resize(visualMap,(512,512))
        visualMap2 = cv2.resize(visualMap2,(512,512))
        predict = cv2.resize(predict,(512,512))
        predict = cv2.cvtColor(predict,cv2.COLOR_GRAY2RGB)
        frame = cv2.vconcat([frame,predict])
        visualMap = cv2.vconcat([visualMap,visualMap2])
        frame = cv2.hconcat([frame,visualMap])
        cv2.imshow("logData",frame)
        # cv2.imwrite("./his/"+str(counter)+".png",frame)
        # print(counter," history")
        returnDict["frameIndex"] = counter

        counter += 1
        # print("after process time:",time.time()-initTime)
        ack = json.dumps(returnDict)
        client.sendall(bytes(str(ack), "utf8"))

        if cv2.waitKey(1) == 27:
            print("Exit")
            release_cap()
            client.sendall(bytes("EXIT", "utf8"))
            cv2.waitKey(100)
            break
        # print("done the loop:",time.time()-initTime)

    # except Exception as e:
    #     print(str(e))
    #     client.close()
    #     f.close()
    #     print("Close connect !")
    #     break  
  

s.close()
print("Shutdown server !")