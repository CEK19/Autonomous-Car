import re
import matplotlib.pyplot as plt

file = open("progress.txt", "r+") 
lines = file.readlines()

def getEpisode(regex):
    return int(regex[0])

def getX(regex):
    return float(regex[1])

def getY(regex):
    return float(regex[2])

def getTotalActions(regex):
    return int(regex[3])

def getTotalReward(regex):
    return float(regex[4])


episodeList = []
xPosList = []
yPosList = []
totalActionsList = []
totalRewardList = []

for idx, line in enumerate(lines):    
    regex = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    
    episode = getEpisode(regex)
    xPos = getX(regex)
    yPos = getY(regex)
    totalAction = getTotalActions(regex)
    totalReward = getTotalReward(regex)
    
    episodeList.append(episode)
    xPosList.append(xPos)
    yPosList.append(yPos)
    totalActionsList.append(totalAction)
    totalRewardList.append(totalReward)        

file.close() 
fig, axs = plt.subplots(2, 2)
fig.set_size_inches(18.5, 8.5)
axs[0, 0].plot(episodeList, xPosList)
axs[0, 0].set_title('xPos per episode')
axs[0, 1].plot(episodeList, yPosList, 'tab:orange')
axs[0, 1].set_title('yPos per episode')
axs[1, 0].plot(episodeList, totalActionsList, 'tab:green')
axs[1, 0].set_title('Total actions per episode')
axs[1, 1].plot(episodeList, totalRewardList, 'tab:red')
axs[1, 1].set_title('Total rewards per episode')

fig.tight_layout()
    
plt.show()