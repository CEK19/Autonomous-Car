import re
import matplotlib.pyplot as plt

file = open("progress.txt", "r+") 
lines = file.readlines()

def getEpisode(regex):
    return int(regex[0])

def getTotalActions(regex):
    return int(regex[1])

def getTotalReward(regex):
    return float(regex[2])


episodeList = []
xPosList = []
yPosList = []
totalActionsList = []
totalRewardList = []

for line in lines:
    regex = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    
    episode = getEpisode(regex)
    totalAction = getTotalActions(regex)
    totalReward = getTotalReward(regex)
    
    episodeList.append(episode)
    totalActionsList.append(totalAction)
    totalRewardList.append(totalReward)
        
file.close() 

fig, axs = plt.subplots(1, 2)
axs[0].plot(episodeList, totalActionsList, 'tab:green')
axs[0].set_title('Total actions per episode')
axs[1].plot(episodeList, totalRewardList, 'tab:red')
axs[1].set_title('Total rewards per episode')

fig.tight_layout()
    
plt.show()