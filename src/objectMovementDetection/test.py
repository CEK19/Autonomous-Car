def getReward(currState, currPlayer):
        
        finalReward = 0
        stateArr = [char for char in currState]
        lidarStates = currPlayer.rayCastingLists
        centerState = stateArr[RLParam.AREA_RAY_CASTING_NUMBERS]

        # Obstacles block car
        # tmp = 0
        for i in range(len(lidarStates)):
            if lidarStates[i] != PlayerParam.INFINITY:
                finalReward -= RLParam.RAY_WEIGHT[i]*(PlayerParam.RADIUS_LIDAR - lidarStates[i])/25
        
        # print(tmp)


        # Car out of lane
        if centerState == RLParam.LEVEL_OF_LANE.MIDDLE:
            finalReward += 20
        elif centerState == RLParam.LEVEL_OF_LANE.RIGHT or centerState == RLParam.LEVEL_OF_LANE.LEFT:
            finalReward += -10
        elif centerState == RLParam.LEVEL_OF_LANE.MOST_RIGHT or centerState == RLParam.LEVEL_OF_LANE.MOST_LEFT:
            finalReward += -100


        # Prevent stop and go back action
        y_Ver = math.cos(currPlayer.currAngle)*currPlayer.currVelocity
        finalReward += -0.5*y_Ver
        if (180-abs(math.degrees(currPlayer.currAngle)%360 - 180)) > 90:
            finalReward += (180-abs(math.degrees(currPlayer.currAngle)%360 - 180))/18-5
        else:
            finalReward += RLParam.GO_FUCKING_DEAD

        if currPlayer.checkCollision():
            finalReward += RLParam.GO_FUCKING_DEAD

        if currPlayer.xPos < 0 or  currPlayer.xPos > GameSettingParam.WIDTH:
            finalReward += RLParam.GO_FUCKING_DEAD

        if currPlayer.yPos > GameSettingParam.HEIGHT:
            finalReward += RLParam.GO_FUCKING_DEAD

        # comment += str(finalReward-lastReward) + "\n"
        # progressFile.write(comment)
            
        return finalReward