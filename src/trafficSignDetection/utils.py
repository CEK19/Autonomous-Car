import os

class Utils:
    @staticmethod
    def getNumOfDataset(path):
        return len(os.listdir(path))