import cv2
import os

# Define input folder and output video file name
input_folder = './data'
output_video = 'merged_video.avi'

# Get list of image file names in input folder
lframe = os.listdir(input_folder)
lframe.sort(key=len)
# Get first image dimensions
img = cv2.imread(os.path.join(input_folder, lframe[0]))
h, w, c = img.shape

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (w, h))

# Loop through image files and add to video writer
for file_name in lframe:
    img = cv2.imread(os.path.join(input_folder, file_name))
    out.write(img)

# Release video writer
out.release()