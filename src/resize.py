import cv2

img = cv2.imread('/home/minhtu/Desktop/hola.png')

img = cv2.resize(img, (128,128))

cv2.imwrite('/home/minhtu/Desktop/hola2.png', img)