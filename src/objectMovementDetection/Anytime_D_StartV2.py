"""
Anytime_D_star 2D
@author: Minh The Tus
"""

import time
import numpy as np
import math
import matplotlib.pyplot as plt
from plottingV2 import *
from const import *
from utils import *
from temp import *


class ADStar:
    def __init__(self, start, goal, eps, heuristic_type, maps):

        # self.Env.motions  # feasible input set
        self.envMotions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                           (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.maps = self.get_map(maps)
        self.envWidth = len(self.maps[0])    # self.Env.width
        self.envHeight = len(self.maps)      # self.Env.height
        Utils.print("width: ", self.envWidth)
        Utils.print("height: ", self.envHeight)

        # self.Env.obs  # position of obstacles
        self.obs = self.obs_map(self.maps)
        Utils.print("obs: ", len(self.obs))
        
        Utils.print("start: ", start)
        Utils.print("goal: ", goal)
        Utils.print("eps: ", eps)
        self.start = self.convertPointPGtoPLT(start)
        self.goal = self.convertPointPGtoPLT(goal)
        self.heuristic_type = heuristic_type

        self.Plot = Plotting(self.start, self.goal, self.obs)
        self.g, self.rhs, self.OPEN = {}, {}, {}

        for i in range(1, self.envWidth - 1):
            for j in range(1, self.envHeight - 1):
                self.rhs[(i, j)] = float("inf")
                self.g[(i, j)] = float("inf")

        self.rhs[self.goal] = 0.0
        self.eps = eps
        self.OPEN[self.goal] = self.Key(self.goal)
        self.CLOSED, self.INCONS = set(), dict()

        self.visited = set()
        self.count = 0
        self.count_env_change = 0
        self.obs_add = set()
        self.obs_remove = set()
        # Anytime D*: Significant changes || Anytime D*: Small changes
        self.title = "Anytime D*: Small changes"
        self.fig = Plotting.createFig()
        
    
    '''
    In PyGame (PG): Oxy is at the top-left corner
    In Matplotlib (PLT): Oxy is at the bottom-left corner
    '''
    def convertPointPGtoPLT(self, point):
        newPoint = (point[0] + 1, point[1] + 1)
        return (newPoint[0], self.envHeight - newPoint[1] - 1)
    def convertPointPLTtoPG(self, point):
        newPoint = (point[0], self.envHeight - point[1] - 1)
        return (newPoint[0] - 1, newPoint[1] - 1)
    
    
    def get_map(self, maps):
        width = len(maps[0]) + 2
        height = len(maps) + 2
        newMaps = [[0 for x in range(width)] for y in range(height)] 
        for rowIndex in range(height):
            for colIndex in range(width):
                if (colIndex == 0 or colIndex == width - 1) or (rowIndex == 0 or rowIndex == height - 1):
                   newMaps[rowIndex][colIndex] = 1
                else:
                    newMaps[rowIndex][colIndex] = maps[rowIndex - 1][colIndex - 1]
        return np.flip(newMaps, axis=0)


    def obs_map(self, maps):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """
        obs = set()

        for rowIndex in range(self.envHeight):
            for colIndex in range(self.envWidth):
                if maps[rowIndex][colIndex] == D_STAR.ENV.HAS_OBS:
                    obs.add((colIndex, rowIndex))
        
        return obs


    def run(self):
        self.ComputeOrImprovePath()
        self.Plot.plot_grid(self.title)
        self.plot_visited()
        self.plot_path(self.extract_path())
        self.visited = set()

        while True:
            if self.eps <= D_STAR.ENV.EPS_MIN:
                break
            self.eps -= D_STAR.ENV.EPS_MINUS_PER_RUN
            self.OPEN.update(self.INCONS)
            for s in self.OPEN:
                self.OPEN[s] = self.Key(s)
            self.CLOSED = set()
            self.ComputeOrImprovePath()
            self.plot_visited()
            self.plot_path(self.extract_path())
            self.visited = set()
            Plotting.pause(0.5)

        # self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        # self.fig.canvas.mpl_connect('button_press_event', self.onChange(D_STAR.NEW_MAP))
        Plotting.show(block=False)


    def onChange(self, newMapInput):
        
        newMap = self.get_map(newMapInput)
        
        tim1 = time.time()
        for rowIndex in range(self.envHeight):
            for colIndex in range(self.envWidth):
                if newMap[rowIndex][colIndex] == D_STAR.ENV.HAS_OBS and (colIndex, rowIndex) not in self.obs:
                    self.obs.add((colIndex, rowIndex))
                    self.obs_add.add((colIndex, rowIndex))
                    Plotting.plot_point(colIndex, rowIndex, 'addObs')
                elif newMap[rowIndex][colIndex] == D_STAR.ENV.NO_OBS and (colIndex, rowIndex) in self.obs:
                    self.obs.remove((colIndex, rowIndex))
                    self.obs_remove.add((colIndex, rowIndex))
                    Plotting.plot_point(colIndex, rowIndex, 'removeObs')
        tim2 = time.time()
        Utils.print("Time to update obs: ", tim2 - tim1)
        
        Utils.print()
        self.Plot.update_obs(self.obs)
        
        # self.eps += 2.0
        self.eps += D_STAR.ENV.EPS_PLUS_PER_ENV_CHANGE
        
        tim3 = time.time()
        Utils.print("len add: ", len(self.obs_add))
        Utils.print("len remove: ", len(self.obs_remove))
        for obs in self.obs_add:
            # self.g[(x, y)] = float("inf")
            # self.rhs[(x, y)] = float("inf")
            for neighbor in self.get_neighbor(obs):
                self.UpdateState(neighbor)

        for obs in self.obs_remove:
            for neighbor in self.get_neighbor(obs):
                self.UpdateState(neighbor)
            self.UpdateState(obs)
        tim4 = time.time()
        Utils.print("Time to update state: ", tim4 - tim3)
        
        Plotting.clearAll();
        self.Plot.plot_grid(self.title)
        
        tim5 = time.time() 
        while True:
            if self.eps <= D_STAR.ENV.EPS_MIN:
                break
            self.eps -= D_STAR.ENV.EPS_MINUS_PER_RUN
            self.OPEN.update(self.INCONS)
            for s in self.OPEN:
                self.OPEN[s] = self.Key(s)
            self.CLOSED = set()
            self.ComputeOrImprovePath()
            self.plot_visited()
            self.plot_path(self.extract_path())
            Plotting.setTitle(self.title)
            self.visited = set()
            Plotting.pause(0.5)
        tim6 = time.time()
        Utils.print("main loop", tim6 - tim5)
        
        Plotting.draw_idle(self.fig)


    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.envWidth - 1 or y < 0 or y > self.envHeight - 1:
            Utils.print("Please choose right area!")
            return

        self.count_env_change += 1
        x, y = int(x), int(y)
        Utils.print("Change position: x =", x, ",y =", y)

        if (x, y) not in self.obs:
            self.obs.add((x, y))
            self.g[(x, y)] = float("inf")
            self.rhs[(x, y)] = float("inf")
        else:
            self.obs.remove((x, y))
            self.UpdateState((x, y))

        self.Plot.update_obs(self.obs)

        for sn in self.get_neighbor((x, y)):
            self.UpdateState(sn)

        Plotting.clearAll()
        self.Plot.plot_grid(self.title)

        while True:
            if len(self.INCONS) == 0:
                Utils.print("INCONS is empty!")
                break
            self.OPEN.update(self.INCONS)
            for s in self.OPEN:
                self.OPEN[s] = self.Key(s)
            self.CLOSED = set()
            self.ComputeOrImprovePath()
            self.plot_visited()
            self.plot_path(self.extract_path())
            Plotting.setTitle(self.title)
            self.visited = set()

            if self.eps <= D_STAR.ENV.EPS_MIN:
                break

        Plotting.draw_idle(self.fig)

    def ComputeOrImprovePath(self):
        Utils.print("compute or improve path")
        tt = 0
        count = 0
        hola = 0
        while True:
            s, v = self.TopKey()
            tt += 1
            
            if s == None and v == None:
                break
            
            if v >= self.Key(self.start) and \
                    self.rhs[self.start] == self.g[self.start]:
                break

            self.OPEN.pop(s)
            self.visited.add(s)

            if self.g[s] > self.rhs[s]:
                count += 1
                self.g[s] = self.rhs[s]
                self.CLOSED.add(s)
                for sn in self.get_neighbor(s):
                    self.UpdateState(sn)
            else:
                hola += 1
                self.g[s] = float("inf")
                for sn in self.get_neighbor(s):
                    self.UpdateState(sn)
                self.UpdateState(s)
        
        Utils.print("tt: ", tt)
        Utils.print("count: ", count)
        Utils.print("hola: ", hola)
        Utils.print()


    def UpdateState(self, curPoint):
        if curPoint != self.goal:
            self.rhs[curPoint] = float("inf")
            for neighbor in self.get_neighbor(curPoint):
                self.rhs[curPoint] = min(self.rhs[curPoint], self.g[neighbor] + self.cost(curPoint, neighbor))
        if curPoint in self.OPEN:
            self.OPEN.pop(curPoint)

        if self.g[curPoint] != self.rhs[curPoint]:
            if curPoint not in self.CLOSED:
                self.OPEN[curPoint] = self.Key(curPoint)
            else:
                self.INCONS[curPoint] = 0


    def Key(self, curPoint):
        if self.g[curPoint] > self.rhs[curPoint]:
            return [self.rhs[curPoint] + self.eps * self.distance(self.start, curPoint), self.rhs[curPoint]]
        else:
            return [self.g[curPoint] + self.distance(self.start, curPoint), self.g[curPoint]]


    def TopKey(self):
        """
        :return: return the min key and its value.
        """
        if (len(self.OPEN) == 0):
            return None, None
        s = min(self.OPEN, key=self.OPEN.get)
        return s, self.OPEN[s]


    def distance(self, start, goal):
        '''
        return: distance between 2 points
        '''
        heuristic_type = self.heuristic_type  # heuristic type

        if heuristic_type == "manhattan":
            return abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        else:
            return math.hypot(goal[0] - start[0], goal[1] - start[1])


    def cost(self, start, goal):
        """
        Calculate Cost for this motion
        :param start: starting node
        :param goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """
        if self.is_collision(start, goal):
            return float("inf")

        return math.hypot(goal[0] - start[0], goal[1] - start[1])


    def is_collision(self, start, end):
        if start in self.obs or end in self.obs:
            return True

        if start[0] != end[0] and start[1] != end[1]:
            if end[0] - start[0] == start[1] - end[1]:
                s1 = (min(start[0], end[0]), min(start[1], end[1]))
                s2 = (max(start[0], end[0]), max(start[1], end[1]))
            else:
                s1 = (min(start[0], end[0]), max(start[1], end[1]))
                s2 = (max(start[0], end[0]), min(start[1], end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False


    def get_neighbor(self, curPoint):
        nei_list = set()
        '''
        envMotions: direction of moving from LEFT -> TOP_LEFT -> TOP -> TOP_RIGHT -> RIGHT -> BOTTOM_RIGHT -> BOTTOM -> BOTTOM_LEFT
        return: list of availabel neighbor points
        '''
        for direction in self.envMotions:
            point = tuple([curPoint[i] + direction[i] for i in range(2)])
            if point[0] < 0 or point[1] < 0 or point[0] >= self.envWidth or point[1] >= self.envHeight:
                continue
            if point not in self.obs:
                nei_list.add(point)

        return nei_list


    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.start]
        curPoint = self.start

        for k in range(100):
            g_list = {}
            for neighbor in self.get_neighbor(curPoint):
                if not self.is_collision(curPoint, neighbor):
                    g_list[neighbor] = self.g[neighbor]
            if not g_list:
                return list()
            curPoint = min(g_list, key=g_list.get)
            path.append(curPoint)
            if curPoint == self.goal:
                break

        return list(path)
    
    
    def getPath(self):
        path = self.extract_path()
        return [self.convertPointPLTtoPG(point) for point in path]


    def plot_path(self, path):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2)
        plt.plot(self.start[0], self.start[1], "bs")
        plt.plot(self.goal[0], self.goal[1], "gs")


    def plot_visited(self):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        
        self.count += 1

        color = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']

        if self.count >= len(color) - 1:
            self.count = 0

        for x in self.visited:
            plt.plot(x[0], x[1], marker='s', color=color[self.count])



class DStarService:
    def __init__(self, start, goal, map):
        self.start = start
        self.goal = goal
        self.heuristicType = "euclidean"
        self.eps = D_STAR.ENV.EPSILON
        self.dStar = ADStar(start, goal, self.eps, self.heuristicType, map)
        self.dStar.run()
    
    def getPath(self):
        return self.dStar.getPath()
    
    def onChange(self, newMap):
        '''
        + change map without start or end point
        '''
        self.dStar = ADStar(self.start, self.goal, self.eps, self.heuristicType, newMap)
        self.dStar.run()
    
    def onReset(self, start, goal, newMap):
        '''
        + change both start, end points and map
        '''
        self.dStar = ADStar(start, goal, self.eps, self.heuristicType, newMap)
        self.dStar.run()


def main():
    # Read data from const file
    start = Gen.genPoint()
    goal = Gen.genPoint()
    
    start1 = Gen.genPoint()
    goal1 = Gen.genPoint()
    
    t1 = time.time()
    map1 = Gen.genMap()
    t2 = time.time()
    map2 = Gen.genMap()
    map3 = Gen.genMap()
    Utils.print("gen map: ", t2 - t1)

    startTime = time.time()
    Utils.print("---init---")
    dstar = DStarService(start, goal, map1)
    endTime1 = time.time()
    
    dstar.getPath()
    endTime2 = time.time()
    
    
    # new value of map
    Utils.print("---change---")
    dstar.onChange(map2)
    endTime3 = time.time()
    
    dstar.getPath()
    endTime4 = time.time()
    
    
    dstar.onReset(start1, goal1, map3)
    endTime5 = time.time()
    
    dstar.getPath()
    endTime6 = time.time()

    
    
    Utils.print()
    Utils.print("init: ", endTime1 - startTime)
    Utils.print("first getPath: ", endTime2 - endTime1)
    Utils.print("change: ", endTime3 - endTime2)
    Utils.print("second getPath: ", endTime4 - endTime3)
    Utils.print("change both map and start end points: ", endTime5 - endTime4)
    Utils.print("third getPath: ", endTime6 - endTime5)
    Utils.print("-----------------------------")
    Utils.print("total time: ", endTime6 - startTime)
    # plt.show()


if __name__ == '__main__':
    main()
    # print()
