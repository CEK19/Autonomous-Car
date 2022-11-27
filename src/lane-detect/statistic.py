import re
import matplotlib.pyplot as plt

filename1 =  "nhan-backend.txt"
filename2 = "thinh-backend.txt"

data1 = []
data2 = []

with open(filename1,'r') as file:
    for line in file:
        i = float(line.split(":")[1])
        data1.append(i)
            
with open(filename2,'r') as file:
    for line in file:
        i = float(line.split(":")[1])
        data2.append(i)

fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
axs[0].scatter(range(len(data1)), data1, label="nhan", linewidths = 1, c ="green",  s =20)
axs[0].legend()
axs[1].scatter(range(len(data2)), data2, label="thinh", linewidths = 1, c ="blue", s=  20)
axs[1].legend()
fig.suptitle('Categorical Plotting')

plt.show()