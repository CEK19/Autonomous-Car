import numpy as np
from utils import *
import cv2
from PIL import Image


img = Image.open('../assets/demo/yellow1.jpg')

# Tăng độ tương phản
output = adjust_contrast(img, 5)
output.show("increase contrast")

# Giảm độ tương phản
output1 = adjust_contrast(img, 0.1)
output1.show("descrease contrast")


cv2.waitKey(0)