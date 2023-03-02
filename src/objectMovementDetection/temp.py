from const import *
import numpy as np

class Gen:
    @staticmethod
    def genMap():
        newMaps = [[0 for x in range(GAME_SETTING.SCREEN_WIDTH)] for y in range(GAME_SETTING.SCREEN_HEIGHT)] 
        for rowIndex in range(GAME_SETTING.SCREEN_HEIGHT):
            for colIndex in range(GAME_SETTING.SCREEN_WIDTH):
                newMaps[rowIndex][colIndex] = np.random.choice([0, 1], p=[0.9, 0.1])
        return np.array(newMaps)
    
    @staticmethod
    def genEmptyMap():
        newMaps = [[0 for x in range(GAME_SETTING.SCREEN_WIDTH)] for y in range(GAME_SETTING.SCREEN_HEIGHT)] 
        return np.array(newMaps)
    
    @staticmethod
    def genPoint():
        return (np.random.randint(1, GAME_SETTING.SCREEN_WIDTH - 2), np.random.randint(1, GAME_SETTING.SCREEN_HEIGHT - 2))
    
    @staticmethod
    def genEmptyMapText():
        file = open('./temp.txt', 'a')
        file.write("[\n")
        for rowCount in range(GAME_SETTING.SCREEN_HEIGHT):
            file.write("\t[")
            for colCount in range(GAME_SETTING.SCREEN_WIDTH):
                if colCount == GAME_SETTING.SCREEN_WIDTH - 1:
                    file.write("0")
                else:
                    file.write('0, ')
                    
            if rowCount == GAME_SETTING.SCREEN_HEIGHT - 1:
                file.write("]\n")
            else:
                file.write("],\n")
        file.write("]\n")

Gen.genEmptyMap()