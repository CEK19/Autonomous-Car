import cv2
import numpy as np
import sys
import os

listfRrame = os.listdir("./data")
listfRrame.sort(key=len)

for eachName in listfRrame:
    image = cv2.imread("./data/"+eachName)

    # Convert the image from BGR to HSV color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the white color in HSV
    lower_white = np.array([0, 0, 70])
    upper_white = np.array([255, 60, 255])

    # Apply the color filter to extract white regions
    white_mask = cv2.inRange(hsv_img, lower_white, upper_white)

    cv2.imshow("mask",white_mask)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the minimum area ellipse for the largest contour
    ellipse = cv2.fitEllipse(largest_contour)

    cv2.ellipse(image, ellipse, (0, 0, 255), 1)

    # Get the center coordinates and axis lengths of the ellipse
    center, axes, angle = ellipse


    # Calculate the endpoints of the major axis using trigonometry
    major_axis_length = max(axes)
    minor_axis_length = min(axes)
    angle_rad = (-angle+90) * np.pi / 180
    x1 = int(center[0] + major_axis_length/2 * np.cos(angle_rad))
    y1 = int(center[1] - major_axis_length/2 * np.sin(angle_rad))
    x2 = int(center[0] - major_axis_length/2 * np.cos(angle_rad))
    y2 = int(center[1] + major_axis_length/2 * np.sin(angle_rad))

    # Draw the line along the major axis
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    h,w= image.shape[:2]
    image = cv2.resize(image,(w*3,h*3))

    # Display the image
    cv2.imshow('Image', image)
    
    if 27 == cv2.waitKey(100):
        break