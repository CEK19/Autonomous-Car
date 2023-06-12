import cv2
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt
import re
import sys
import seaborn as sns


nhanf = open("nhan-time.txt", "r")
lines = nhanf.readlines()
nhanData = []

for line in lines:
    lt = re.findall('-?\d+\.?\d*',line)
    if float(lt[0]) > 0.01:
        print("-")
        continue
    nhanData.append(float(lt[0]))
nhanf.close()

thinhf = open("thinh-time.txt", "r")
lines = thinhf.readlines()
thinhData = []

for line in lines:
    lt = re.findall('-?\d+\.?\d*',line)
    thinhData.append(float(lt[0]))
thinhf.close()

# plt.plot(range(1,110),listData)
# sns.displot(thinhf, kind="kde")
plt.violinplot([thinhData,nhanData],showmedians=True)
# plt.violinplot(nhanData)
plt.ylabel("Thời gian")
plt.title("So sánh Thời gian thực thi hai backend")
plt.legend()
plt.show()

