import numpy as np
from PIL import Image, ImageEnhance
import cv2

#faster way to compute
#reference: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
# gamma < 1 => tăng tương phản
# gamma > 1 => giảm tương phản
# def adjust_image_gamma_lookuptable(image, gamma=1.0):
# 	# build a lookup table mapping the pixel values [0, 255] to
# 	# their adjusted gamma values
# 	table = np.array([((i / 255.0) ** gamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")

# 	# apply gamma correction using the lookup table
# 	return cv2.LUT(image, table)


def adjust_contrast(image, factor):
	enhancer = ImageEnhance.Contrast(image)
	return enhancer.enhance(factor)

def adjust_brightness(image, factor):
	enhancer = ImageEnhance.Brightness(image)
	return enhancer.enhance(factor)
