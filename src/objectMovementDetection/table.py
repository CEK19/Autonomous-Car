from const import *


class RLAlgorithm:
    def __init__(self, rayCastingData, actions) -> None:
        # Converting n raycasting to signal in area , min raycast of each area
        self.signalPerAreaData = self._initSignalPerArea(rayCastingData=rayCastingData)
                
        self.Q = self._initQTable(actions=actions)
        
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

    def train(self):
        pass
        
        
RL = RLAlgorithm(range(90), [])
print(RL.signalPerAreaData)