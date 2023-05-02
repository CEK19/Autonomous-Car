from ultralytics import YOLO
import cv2
import numpy as np
# https://docs.ultralytics.com/modes/predict/#boxes
img = cv2.imread("/Users/mac/Desktop/test/object_detection/images/lvtn_4_mix/images/v1_frame_0300.jpg")
model = YOLO("/Users/mac/Desktop/test/object_detection/results/content/runs/detect/train/weights/last.pt")
# model.predict(source="/Users/mac/Desktop/test/object_detection/video/stop.avi", show=True, conf=0.8)
results = model([img])
# print((results[0].boxes.xywh[0][0].item()))
# mask = np.zeros(image.shape[:2], dtype=np.uint8)