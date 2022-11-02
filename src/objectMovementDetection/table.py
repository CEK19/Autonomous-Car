from const import *
import random
import numpy as np

class RLAlgorithm:
    def __init__(self, rayCastingData, actions) -> None:
        # Converting n raycasting to signal in area , min raycast of each area
        self.signalPerAreaData = self._initSignalPerArea(rayCastingData=rayCastingData)
                
        self.Q = self._initQTable(actions=actions)
        self.actions = actions
        
    def _initSignalPerArea(self, rayCastingData):
        div = len(rayCastingData) // RLParam.AREA_RAY_CASTING_NUMBERS
        mod = len(rayCastingData) % RLParam.AREA_RAY_CASTING_NUMBERS

        tmpData = [0] * (RLParam.AREA_RAY_CASTING_NUMBERS +
                            (0 if mod == 0 else 1))

        tmpCount = 0
        for i in range(len(tmpData)):
            tmp = []
            for _ in range(div):
                tmp.append(rayCastingData[tmpCount])
                tmpCount += 1
                if (tmpCount == len(rayCastingData)):
                    break
            tmpData[i] = min(tmp)
        return tmpData        

    def _initQTable(self, actions):
        pass
    
    def _hashFromDistanceToState(self): # Tu
        pass
    
    def getReward(currState, currAction): 
        pass
    
    def _epsilonGreedyPolicy(self, currState):
        if random.uniform(0, 1) < RLParam.EPSILON:
            return random.choice(range(len(self.actions)))
        else:
            return np.argmax(self.Q[currState])
        

    def train(self):
        
        alphas = np.linspace(RLParam.MAX_ALPHA, RLParam.MIN_ALPHA, RLParam.N_EPISODES)
        
        for e in range(RLParam.N_EPISODES):
            # TODO: write get initState
            state = "START STATE"
            totalReward = 0
            alpha = alphas[e]
            
            for _ in range(RLParam.MAX_EPISODE_STEPS):
                actionIndex = self._epsilonGreedyPolicy(currState=state)
                nextState, reward, done = act(state, actionIndex)
                totalReward += reward
                self.Q[state][actionIndex] = self.Q[state][actionIndex] + \
                        alpha * (reward + RLParam.GAMMA *  np.max(self.Q[nextState]) - self.Q[state][actionIndex])
                state = nextState
                if done:
                    break
            print(f"Episode {e + 1}: total reward -> {totalReward}")        
        
        
RL = RLAlgorithm(range(90), [1, 2, 3, 4])
print(RL.signalPerAreaData)