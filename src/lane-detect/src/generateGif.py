import imageio
import os
images = []
filenames = os.listdir("D:/container/AI_DCLV/readData/video/outputImg/st1/final")
filenames.sort(key = len)
# print(filenames)
c = 0
for filename in filenames:
    if c == 0:
        c = 1
        continue
    else:
        c = 0
    print(filename)
    images.append(imageio.imread("D:/container/AI_DCLV/readData/video/outputImg/st1/final/"+filename))
imageio.mimsave('final.gif', images)