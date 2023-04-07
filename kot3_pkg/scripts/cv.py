import cv2
import numpy as np

map = np.zeros((100, 400))
cv2.line(map, (-10, -10), (200, 200), 255, 1)

cv2.imshow("map", map)

cv2.waitKey(0)