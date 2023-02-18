# múa lửa cho Nhân dui, Nhân cọc quá :<<
import numpy as np
from utils import *
import cv2
from PIL import Image

# img = cv2.imread('./assets/demo/red1.jpg')
# cv2.imshow("orgImg", img)

# blurImg = cv2.GaussianBlur(img, (13, 13), 0)
# cv2.imshow("blurredImg", blurImg)

img = Image.open('./assets/demo/yellow1.jpg')

output = adjust_contrast(img, 5)
output.show("increase contrast")
# cv2.imshow("increase contrast", output)

output1 = adjust_contrast(img, 0.1)
output1.show("descrease contrast")
# cv2.imshow("descrease contrast", output1)


cv2.waitKey(0)