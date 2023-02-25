"""
Anytime_D_star 2D
@author: Minh The Tus
"""

import time
import numpy as np
import math
import matplotlib.pyplot as plt
import plottingV2 as plotting
from const import *


class ADStar:
    def __init__(self, start, goal, eps, heuristic_type, maps):

        # self.Env.motions  # feasible input set
        self.envMotions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                           (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.maps = np.flip(maps, axis=0)
        self.envWidth = len(self.maps[0])    # self.Env.width
        self.envHeight = len(self.maps)      # self.Env.height
        print("width: ", self.envWidth)
        print("height: ", self.envHeight)

        # self.Env.obs  # position of obstacles
        self.obs = self.obs_map(self.maps)
        print("obs: ", self.obs)
        
        print("start: ", start)
        print("goal: ", goal)
        print("eps: ", eps)
        self.start = self.convertPointPGtoPLT(start)
        self.goal = self.convertPointPGtoPLT(goal)
        self.heuristic_type = heuristic_type

        self.Plot = plotting.Plotting(self.start, self.goal, self.obs)
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
        self.fig = self.Plot.createFig()
        
    
    '''
    In PyGame (PG): Oxy is at the top-left corner
    In Matplotlib (PLT): Oxy is at the bottom-left corner
    '''
    def convertPointPGtoPLT(self, point):
        return (point[0], self.envHeight - point[1] - 1)
    def convertPointPLTtoPG(self, point):
        return (point[0], self.envHeight - point[1] - 1)


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
            self.Plot.pause(0.5)

        # self.Plot.pause(3)
        # self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        # self.fig.canvas.mpl_connect('button_press_event', self.onChange(D_STAR.NEW_MAP))
        self.Plot.show(block=False)


    def onChange(self, newMapInput):
        
        newMap = np.flip(newMapInput, axis=0)
        
        for rowIndex in range(self.envHeight):
            for colIndex in range(self.envWidth):
                if newMap[rowIndex][colIndex] == D_STAR.ENV.HAS_OBS and (colIndex, rowIndex) not in self.obs:
                    self.obs.add((colIndex, rowIndex))
                    self.obs_add.add((colIndex, rowIndex))
                    self.Plot.plot_point(colIndex, rowIndex, 'addObs')
                elif newMap[rowIndex][colIndex] == D_STAR.ENV.NO_OBS and (colIndex, rowIndex) in self.obs:
                    self.obs.remove((colIndex, rowIndex))
                    self.obs_remove.add((colIndex, rowIndex))
                    self.Plot.plot_point(colIndex, rowIndex, 'removeObs')
        
        print()
        print("obs_add: ", self.obs_add)
        print("obs_remove: ", self.obs_remove)
        self.Plot.update_obs(self.obs)
        
        # self.eps += 2.0
        self.eps += D_STAR.ENV.EPS_PLUS_PER_ENV_CHANGE
        
        for obs in self.obs_add:
            # self.g[(x, y)] = float("inf")
            # self.rhs[(x, y)] = float("inf")
            for neighbor in self.get_neighbor(obs):
                self.UpdateState(neighbor)

        for obs in self.obs_remove:
            for neighbor in self.get_neighbor(obs):
                self.UpdateState(neighbor)
            self.UpdateState(obs)
        
        self.Plot.clearAll();
        self.Plot.plot_grid(self.title)
                    
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
            self.Plot.setTitle(self.title)
            self.visited = set()
            self.Plot.pause(0.5)
        
        self.Plot.draw_idle(self.fig)


    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.envWidth - 1 or y < 0 or y > self.envHeight - 1:
            print("Please choose right area!")
            return

        self.count_env_change += 1
        x, y = int(x), int(y)
        print("Change position: x =", x, ",", "y =", y)
        
        # # for small changes
        # if self.title == "Anytime D*: Small changes":

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

        self.Plot.clearAll()
        self.Plot.plot_grid(self.title)

        while True:
            if len(self.INCONS) == 0:
                print("INCONS is empty!")
                break
            self.OPEN.update(self.INCONS)
            for s in self.OPEN:
                self.OPEN[s] = self.Key(s)
            self.CLOSED = set()
            self.ComputeOrImprovePath()
            self.plot_visited()
            self.plot_path(self.extract_path())
            self.Plot.setTitle(self.title)
            self.visited = set()

            if self.eps <= D_STAR.ENV.EPS_MIN:
                break

        self.Plot.draw_idle(self.fig)

    def ComputeOrImprovePath(self):
        while True:
            s, v = self.TopKey()
            if v >= self.Key(self.start) and \
                    self.rhs[self.start] == self.g[self.start]:
                break

            self.OPEN.pop(s)
            self.visited.add(s)

            if self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                self.CLOSED.add(s)
                for sn in self.get_neighbor(s):
                    self.UpdateState(sn)
            else:
                self.g[s] = float("inf")
                for sn in self.get_neighbor(s):
                    self.UpdateState(sn)
                self.UpdateState(s)
        
        print()
        print("path: ", self.extract_path())
        print()


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
        # print()
        # print("curPoint: ", curPoint)
        # print("self.g: ", self.g)
        # print("self.g[curPoint]: ", self.g[curPoint])
        # print("self.rhs: ", self.rhs)
        # print("self.rhs[curPoint]: ", self.rhs[curPoint])
        # print()
        if self.g[curPoint] > self.rhs[curPoint]:
            return [self.rhs[curPoint] + self.eps * self.distance(self.start, curPoint), self.rhs[curPoint]]
        else:
            return [self.g[curPoint] + self.distance(self.start, curPoint), self.g[curPoint]]


    def TopKey(self):
        """
        :return: return the min key and its value.
        """

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



def main():
    # Read data from const file
    start = D_STAR.ENV.START_POINT
    goal = D_STAR.ENV.GOAL_POINT
    eps = D_STAR.ENV.EPSILON

    dstar = ADStar(start, goal, eps, "euclidean", D_STAR.MY_MAP)
    dstar.run()
    print("getPath: ", dstar.getPath())
    time.sleep(2)
    
    # new value of map
    dstar.onChange(D_STAR.NEW_MAP)
    print("getPath: ", dstar.getPath())
    plt.show()


if __name__ == '__main__':
    main()
