import cv2
import numpy as np
import time

# read image
img = cv2.imread('./assets/tmp/red2_inpaint.png')
cv2.imshow("IMAGE", img)


# find white color in picture
lower = (170,170,170)
upper = (255,255,255)
thresh = cv2.inRange(img, lower, upper)
cv2.imshow("THRESH", thresh)
cv2.imshow("thresh with color", cv2.bitwise_and(img, img, mask=thresh))


# fill these white color area with 4 pixel surrounding color
start = time.time()
result = cv2.inpaint(img, thresh, 4, cv2.INPAINT_TELEA)
end = time.time()
print(end-start, " seconds")
cv2.imshow("RESULT", result)


# save result
cv2.imwrite("./assets/tmp/red2_inpaintV2.png", result)
cv2.waitKey(0)