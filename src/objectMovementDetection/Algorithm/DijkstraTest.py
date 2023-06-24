import cv2
import numpy as np

def BFS_SP(graph, start, goal):
    explored = []
     
    # Queue for traversing the
    # graph in the BFS
    queue = [[start]]
     
    # If the desired node is
    # reached
    if start == goal:
        print("Same Node")
        return
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]
         
        # Condition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = graph[node]
             
            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                 
                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    return new_path
            explored.append(node)
 
    # Condition when the nodes
    # are not connected
    print("So sorry, but a connecting"\
                "path doesn't exist :(")
    return

def pathName(i,j):
    return str(i)+"-"+str(j)

def name2point(name):
    name = name.split("-")
    return [int(name[0]),int(name[1])]

def delGraph(graph,name):
    for each in graph[name]:
        graph[each].remove(name)
    graph[name] = []
    return graph

def delAllGraphInCircle(graph,center,radius):
    for i in range(center[0]-radius,center[0]+radius):
        for j in range(center[1]-radius,center[1]+radius):
            if (radius**2 > (center[0]-i)**2 + (center[1]-j)**2):
                graph = delGraph(graph,pathName(i,j))
    return graph

def findAllNodeInCircle(graph,center,radius):
    data = []
    for i in range(center[0]-radius,center[0]+radius):
        for j in range(center[1]-radius,center[1]+radius):
            if (radius**2 > (center[0]-i)**2 + (center[1]-j)**2):
                data.append(pathName(i,j))
    return graph



size = 100

for eachRun in range(100):
    graph = {}
    for i in range(size):
        for j in range(size):
            graph[pathName(i,j)] = []

    circleRadius = 10
    circleCenter = (np.random.randint(size//2-15,size//2+15),np.random.randint(size//2-15,size//2+15))

    for i in range(size):
        for j in range(size):
            thisPos = pathName(i,j)
            if i>0:
                graph[thisPos].append(pathName(i-1,j))
            if i<size-1:
                graph[thisPos].append(pathName(i+1,j))
            if j>0:
                graph[thisPos].append(pathName(i,j-1))
            if j<size-1:
                graph[thisPos].append(pathName(i,j+1))
            if i>0 and j>0:
                graph[thisPos].append(pathName(i-1,j-1))
            if i<size-1 and j>0:
                graph[thisPos].append(pathName(i+1,j-1))
            if i>0 and j<size-1:
                graph[thisPos].append(pathName(i-1,j+1))
            if i<size-1 and j<size-1:
                graph[thisPos].append(pathName(i+1,j+1))
            

    for i in range(1,size-1):
        graph[pathName(i-1,size//2)].append(pathName(i+1,size//2))
        graph[pathName(i+1,size//2)].append(pathName(i-1,size//2))

    frame = np.zeros((size,size,3),dtype='uint8')

    # circleRadius = np.random.randint(5,20)
    

    print("start del")

    graph = delAllGraphInCircle(graph,circleCenter,circleRadius)
    BFS_path = BFS_SP(graph,pathName(0,50),pathName(99,50))
    frame = cv2.circle(frame,circleCenter,circleRadius,(0,0,255),-1)
    for each in range(len(BFS_path)-1):
        frame = cv2.line(frame,name2point(BFS_path[each]),name2point(BFS_path[each+1]),(0,255,0),1)

    cv2.imshow("frame",frame)
    cv2.imwrite("./someTestCode/DijData/"+str(eachRun)+".png",frame)
    print(eachRun)
    if cv2.waitKey(1) == 27:
        break



# print(graph["10"])
        