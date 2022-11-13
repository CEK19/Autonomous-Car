import re
import matplotlib.pyplot as plt

file = open("y.txt", "r+") 
lines = file.readlines()
print(lines)
yPosList = []
xPosList = []
for idx, line in enumerate(lines):    
    regex = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    
    yPosList.append(float(line))
    xPosList.append(idx)

print(yPosList)
plt.plot(xPosList, yPosList)
plt.show()