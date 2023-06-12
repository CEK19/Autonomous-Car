import numpy as np
import cv2

srcPos = np.float32([(13,71),(113,76),(37,42),(91,43)])
dstPos = np.float32([(16,2),(38,2),(19,-21),(37,-21)])
transformMatrix = cv2.getPerspectiveTransform(srcPos, dstPos)
print(transformMatrix)