import cv2
import numpy as np
import os
import random

def randomize_rect(image, x, y, width, height):
    # Tạo một bản sao của hình ảnh đầu vào để tránh ảnh hưởng đến hình ảnh gốc
    img_copy = np.copy(image)

    # Tạo một hình chữ nhật ngẫu nhiên với kích thước được xác định bởi đầu vào (x,y,width,height)
    rect = img_copy[y:y+height, x:x+width]

    # Tạo các giá trị ngẫu nhiên cho các pixel trong hình chữ nhật
    random_values = np.random.randint(low=0, high=256, size=rect.shape)

    # Gán các giá trị ngẫu nhiên vào các pixel trong hình chữ nhật
    rect[:] = random_values

    # Trả về hình ảnh được cập nhật
    return img_copy

def augment_image(image):
    # Randomly apply Gaussian blur
    blur_radius = random.randint(0, 3)
    if blur_radius > 0:
        image = cv2.GaussianBlur(image, (2 * blur_radius + 1, 2 * blur_radius + 1), 0)

    # Randomly adjust brightness
    brightness = random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

    # Randomly adjust dark
    dark = random.randint(-150, 10)
    lut = np.arange(0, 256, 1, dtype=np.int32)
    lut = np.clip(lut + dark, 0, 255).astype(np.uint8)
    image = cv2.LUT(image, lut)

    # Randomly add frost effect
    frost_level = random.uniform(0, 0.5)
    noise = np.random.randn(*image.shape) * 50
    noise = np.clip(noise, -255, 255).astype(np.uint8)
    image = cv2.addWeighted(image, 1 - frost_level, noise, frost_level, 0)

    # Randomly hide some part of the image
    image = randomize_rect(image,np.random.randint(80),np.random.randint(80),np.random.randint(47),np.random.randint(47))
    return image

for each in range(100):
    img = cv2.imread("./output/ras3-0.png")
    img = augment_image(img)
    cv2.imshow("img",img)
    if cv2.waitKey(0) == 27:
        break
