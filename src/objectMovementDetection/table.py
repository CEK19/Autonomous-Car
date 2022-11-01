from const import *
class RLAlgorithm:
    def __init__(self, rayCastingData, actions) -> None:                
        # Converting n raycasting to signal in area , min raycast of each area        
        div = len(rayCastingData) // RLParam.AREA_RAY_CASTING_NUMBERS
        mod = len(rayCastingData) % RLParam.AREA_RAY_CASTING_NUMBERS
        self.signalPerAreaData = [0] * (RLParam.AREA_RAY_CASTING_NUMBERS + (0 if mod == 0  else  1))        
        
        tmpCount = 0
        for i in range(len(self.signalPerAreaData)):
            tmp = []
            for _ in range(div):
                tmp.append(rayCastingData[tmpCount])
                tmpCount += 1
                if (tmpCount == len(rayCastingData)): 
                    break
            self.signalPerAreaData[i] = min(tmp)
        
        self.Q = {

        }
        
        
RL = RLAlgorithm(range(91), [])
print(RL.signalPerAreaData)