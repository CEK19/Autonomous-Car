from const import *


class RLAlgorithm:
    def __init__(self, rayCastingData, actions) -> None:
        # Converting n raycasting to signal in area , min raycast of each area
        self.signalPerAreaData = self._initSignalPerArea(rayCastingData=rayCastingData)
        self.leftSideDistance = 16
        self.rightSideDistance = 19
                
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
        hashFromRayCasting = ""
        for signal in self.signalPerAreaData:
            for index, distanceRange in enumerate(RLParam.DISTANCE_OF_RAY_CASTING):
                if index == len(RLParam.DISTANCE_OF_RAY_CASTING) - 1:
                    hashFromRayCasting += RLParam.LEVEL_OF_RAY_CASTING.INFINITY
                    break
                elif signal < distanceRange:
                    hashFromRayCasting += str(index)
                    break
        
        hashFromCeneterOfLane = ""
        distanceFromCenterOfLane = abs(self.leftSideDistance - self.rightSideDistance) / 2
        for index, distance in enumerate(RLParam.DISTANCE_FROM_CENTER_OF_LANE):
            if index == len(RLParam.DISTANCE_FROM_CENTER_OF_LANE) - 1:
                hashFromCeneterOfLane += RLParam.LEVEL_OF_LANE.MIDDLE
                break
            elif distanceFromCenterOfLane > distance:
                if self.leftSideDistance < self.rightSideDistance:
                    hashFromCeneterOfLane += str(index + int(RLParam.LEVEL_OF_LANE.MIDDLE) + 1)
                else:
                    hashFromCeneterOfLane += str(index)
                break
        return hashFromRayCasting + hashFromCeneterOfLane
    
    def getReward(self, currState, currAction): 
        stateArr = [int(char) for char in currState]
        lidarState = stateArr[0:RLParam.AREA_RAY_CASTING_NUMBERS]
        centerState = stateArr[-1]
        
        # Obstacles block car
        # if()
        
        print(lidarState, "-", centerState)
        pass

    def train(self):
        pass
        
        
RL = RLAlgorithm(range(90), [])
print(RL.signalPerAreaData)
state = RL._hashFromDistanceToState()
print(state)
RL.getReward(state, 0)