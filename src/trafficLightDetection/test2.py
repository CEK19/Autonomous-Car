import cv2
import numpy as np
import time

# read image
img = cv2.imread('./assets/tmp/red2_inpaint.png')
hh, ww = img.shape[:2]

# threshold
lower = (150,150,150)
upper = (255,255,255)
thresh = cv2.inRange(img, lower, upper)


# apply morphology close and open to make mask
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
# morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=1)

# floodfill the outside with black
# black = np.zeros([hh + 2, ww + 2], np.uint8)
# mask = morph.copy()
# mask = cv2.floodFill(mask, black, (0,0), 0, 0, 0, flags=8)[1]

# use mask with input to do inpainting
# result1 = cv2.inpaint(img, mask, 101, cv2.INPAINT_TELEA)
# result2 = cv2.inpaint(img, mask, 101, cv2.INPAINT_NS)
start = time.time()
result = cv2.inpaint(img, thresh, 4, cv2.INPAINT_TELEA)
end = time.time()
print(end-start, " seconds")

# start = time.time()
# result0 = cv2.inpaint(img, thresh, 4, cv2.INPAINT_NS)
# end = time.time()
# print(end-start, " seconds")

# display it
cv2.imshow("IMAGE", img)

cv2.imshow("THRESH", thresh)
cv2.imshow("thresh with color", cv2.bitwise_and(img, img, mask=thresh))

# cv2.imshow("MORPH", morph)
# cv2.imshow("morph with color", cv2.bitwise_and(img, img, mask=morph))

# cv2.imshow("MASK", mask)
# cv2.imshow("mask with color", cv2.bitwise_and(img, img, mask=mask))

cv2.imshow("RESULT", result)
# cv2.imshow("RESULT0", result0)
# cv2.imshow("RESULT1", result1)
# cv2.imshow("RESULT2", result2)
cv2.imwrite("./assets/tmp/red2_inpaintV2.png", result)
cv2.waitKey(0)