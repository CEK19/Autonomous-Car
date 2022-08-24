import cv2

img = cv2.imread('C:/Users/Admin/Documents/coding/Autonomous-Car/dataset/dataset_kitty/data_road/training/image_2/um_000000.png')

img = cv2.circle(img, (7.215377000000e+02,10), radius=5, color=(0, 0, 255), thickness=-1)

cv2.imshow("ksjl", img)

cv2.waitKey(0)