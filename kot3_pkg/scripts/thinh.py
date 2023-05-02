from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.dijkstra import DijkstraFinder
from pathfinding.finder.best_first import BestFirst
from pathfinding.finder.bi_a_star import BiAStarFinder
from pathfinding.finder.breadth_first import BreadthFirstFinder
from pathfinding.finder.ida_star import IDAStarFinder
from pathfinding.finder.msp import MinimumSpanningTree

import numpy as np
import cv2
import time

for tryTime in range(10):
    map = np.zeros((50,50),dtype='uint8')
    map += 100
    for _ in range(5):
        map = cv2.rectangle(map,(np.random.randint(2,48),np.random.randint(2,48)),(np.random.randint(2,48),np.random.randint(2,48)),0,-1)
    map = np.array(map,dtype='int')

    grid = Grid(matrix=map)
    start = grid.node(0,0)
    end = grid.node(25,48)

    grid = Grid(matrix=map)
    start = grid.node(0,0)
    end = grid.node(25,48)
    startTime = time.time()
    finder2 = BestFirst(diagonal_movement=DiagonalMovement.always)
    path2, runs = finder2.find_path(start, end, grid)
    time2 = time.time()-startTime

    map = np.array(map,dtype='uint8')
    map = cv2.cvtColor(map,cv2.COLOR_GRAY2BGR)
    print("path2", path2)
    for each in path2:
        map[each[1]][each[0]] = [0,0,250]

    map = cv2.resize(map,(250,250),cv2.INTER_LINEAR)

    cv2.imshow("m",map)
    print(time2)
    cv2.waitKey(1000)
# print(path)
# print(grid.grid_str(path=path, start=start, end=end))



'''
AStarFinder 4.0 3.5
DijkstraFinder 5.7 3.5
BestFirst 0.02 0.15
BiAStarFinder 1.21 0.9
BreadthFirstFinder 0.48 0.36
IDAStarFinder 0.003 0.005
MinimumSpanningTree 5.9 3.5
'''
