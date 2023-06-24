import math
import pygame
from sys import exit
from const import *
from utils import *
import random
import numpy as np
from datetime import datetime
import Anytime_D_StartV2 as atdstar
from genGraph import genData
import cv2
from Gen import *


class Lane():
    def __init__(self):
        #  (left, top, width, height)
        self.leftLane = pygame.Rect((LANE_SETTING.LEFT_PADDING, LANE_SETTING.TOP_PADDING), (
            LANE_SETTING.WIDTH_OF_LANE_BORDER, GAME_SETTING.SCREEN_HEIGHT))
        self.rightLane = pygame.Rect((GAME_SETTING.SCREEN_WIDTH - LANE_SETTING.RIGHT_PADDING,
                                      LANE_SETTING.TOP_PADDING), (LANE_SETTING.WIDTH_OF_LANE_BORDER, GAME_SETTING.SCREEN_HEIGHT))

    def draw(self, screen):
        pygame.draw.rect(screen, COLOR.WHITE, self.leftLane)
        pygame.draw.rect(screen, COLOR.WHITE, self.rightLane)

    def isCollideWithThing(self, thing):
        return self.leftLane.colliderect(thing) or self.rightLane.colliderect(thing)


class Car():
    def __init__(self, initX, initY, maxForwardVelocity, minRotationVelocity, maxRotationVelocity, accelerationForward, accelerationRotate, radiusObject) -> None:
        self.xPos, self.yPos = initX, initY
        self.maxForwardVelocity = maxForwardVelocity
        self.maxRotationVelocity = maxRotationVelocity
        self.minRotationVelocity = minRotationVelocity

        self.currentForwardVelocity = 0  # always >= 0
        self.currRotationVelocity = 0  # rotate left < 0, rotate right > 0

        self.currAngle = math.pi
        self.accelerationForward = accelerationForward
        self.accelerationRotate = accelerationRotate

        self.radiusObject = radiusObject

    def move(self, action):
        if action == ACTIONS.FORWARD_ACCELERATION:
            self.currentForwardVelocity = min(
                self.currentForwardVelocity + self.accelerationForward, self.maxForwardVelocity)
        elif action == ACTIONS.BACKWARD_ACCELERATION:
            self.currentForwardVelocity = max(
                self.currentForwardVelocity - self.accelerationForward, 0)
        elif action == ACTIONS.TURN_LEFT_ACCELERATION:
            self.currRotationVelocity = max(
                self.currRotationVelocity - self.accelerationRotate, self.minRotationVelocity)
        elif action == ACTIONS.TURN_RIGHT_ACCELERATION:
            self.currRotationVelocity = min(
                self.currRotationVelocity + self.accelerationRotate, self.maxRotationVelocity)
        elif action == ACTIONS.STOP:
            self.currentForwardVelocity = 0
            self.currRotationVelocity = 0
        elif action == ACTIONS.DO_NOTHING:
            pass

        # Calculate the position base on velocity per frame
        dt = float(1/GAME_SETTING.FPS)

        self.currAngle += (self.currRotationVelocity*dt)

        # Prevent car go to the opposite way
        if (self.currAngle < 0):
            self.currAngle = 2*math.pi - abs(self.currAngle)
        elif (self.currAngle > 2*math.pi):
            self.currAngle = abs(self.currAngle - 2*math.pi)

        # Update the new position based on the velocity
        self.yPos += math.cos(self.currAngle) * \
            self.currentForwardVelocity * dt
        self.xPos += -math.sin(self.currAngle) * \
            self.currentForwardVelocity * dt


class Robot(Car):
    def __init__(self) -> None:
        super().__init__(
            initX=np.random.randint(PLAYER_SETTING.INITIAL_X-50,PLAYER_SETTING.INITIAL_X+50),
            initY=PLAYER_SETTING.INITIAL_Y,
            maxForwardVelocity=PLAYER_SETTING.MAX_FORWARD_VELO,
            minRotationVelocity=PLAYER_SETTING.MIN_ROTATION_VELO,
            maxRotationVelocity=PLAYER_SETTING.MAX_ROTATION_VELO,
            accelerationForward=PLAYER_SETTING.ACCELERATION_FORWARD,
            accelerationRotate=PLAYER_SETTING.ACCELERATION_ROTATE,
            radiusObject=PLAYER_SETTING.RADIUS_OBJECT,
        )
        self.is_alive = True
        self.is_goal = False
        self.currAngle = math.pi + np.random.random()-0.5
        self.lidarSignals = [PLAYER_SETTING.RADIUS_LIDAR]*PLAYER_SETTING.CASTED_RAYS
        self.lidarVisualize = [{"source": {"x": self.xPos, "y": self.yPos},
                                "target": {"x": self.xPos, "y": self.yPos},
                                "color": COLOR.WHITE
                                } for x in range(PLAYER_SETTING.CASTED_RAYS)]
                                

    def checkCollision(self, collisions, lane):
        if collisions == None or len(collisions) == 0:
            self.is_alive = True
            return

        recColliderAroundRobot = pygame.Rect(self.xPos - self.radiusObject, self.yPos -
                                             self.radiusObject, self.radiusObject*2, self.radiusObject*2)
        if lane.isCollideWithThing(recColliderAroundRobot):
            self.is_alive = False
            print(self.is_alive, datetime.now().strftime("%H:%M:%S"))
            return

        for collision in collisions:
            distanceBetweenCenter = Utils.distanceBetweenTwoPoints(
                self.xPos, self.yPos, collision.xPos, collision.yPos)
            if distanceBetweenCenter <= PLAYER_SETTING.RADIUS_OBJECT + OBSTACLE_SETTING.RADIUS_OBJECT:
                self.is_alive = False
                return

        if self.xPos <= 0 or self.xPos >= GAME_SETTING.SCREEN_WIDTH or self.yPos < 0 or self.yPos >= GAME_SETTING.SCREEN_HEIGHT:
            self.is_alive = False
            return

    def scanLidar(self, obstacles):
        inRangeLidarObject = []
        for obstacle in obstacles:
            distance = Utils.distanceBetweenTwoPoints(
                self.xPos, self.yPos, obstacle.xPos, obstacle.yPos)
            isInRageLidar = distance < OBSTACLE_SETTING.RADIUS_OBJECT + \
                PLAYER_SETTING.RADIUS_LIDAR
            if (isInRageLidar == True):
                inRangeLidarObject.append(obstacle)

        startAngle = self.currAngle - PLAYER_SETTING.HALF_FOV

        if (len(inRangeLidarObject) == 0):
            for ray in range(PLAYER_SETTING.CASTED_RAYS):
                target_x = self.xPos - \
                    math.sin(startAngle) * PLAYER_SETTING.RADIUS_LIDAR
                target_y = self.yPos + \
                    math.cos(startAngle) * PLAYER_SETTING.RADIUS_LIDAR
                self.lidarVisualize[ray]["target"] = {
                    "x": target_x,
                    "y": target_y
                }
                self.lidarVisualize[ray]["source"] = {
                    "x": self.xPos,
                    "y": self.yPos
                }
                self.lidarVisualize[ray]["color"] = COLOR.CYAN
                self.lidarSignals[ray] = PLAYER_SETTING.RADIUS_LIDAR
                startAngle += PLAYER_SETTING.STEP_ANGLE
        else:
            for ray in range(PLAYER_SETTING.CASTED_RAYS):
                self.lidarVisualize[ray]["source"] = {
                    "x": self.xPos,
                    "y": self.yPos
                }
                isObjectDetected = False
                target_x = self.xPos - \
                    math.sin(startAngle) * PLAYER_SETTING.RADIUS_LIDAR
                target_y = self.yPos + \
                    math.cos(startAngle) * PLAYER_SETTING.RADIUS_LIDAR

                for obstacle in inRangeLidarObject:
                    # TODO: DOUBLE CHECK WITH THỊNH
                    theda = math.sqrt((obstacle.xPos - self.xPos)
                                      ** 2+(obstacle.yPos - self.yPos)**2)
                    beta = 0

                    if obstacle.xPos - self.xPos < 0:
                        beta = math.acos(
                            (obstacle.yPos - self.yPos)/theda) - startAngle
                    else:
                        beta = math.acos(
                            (obstacle.yPos - self.yPos)/theda) + startAngle
                    height = theda*math.sin(beta)

                    # Calculate scalar: tích vô hướng
                    scalar = (self.xPos-obstacle.xPos)*math.sin(startAngle) + \
                        (obstacle.yPos-self.yPos)*math.cos(startAngle)

                    if abs(height) < PLAYER_SETTING.RADIUS_OBJECT and scalar > 0:
                        isObjectDetected = True

                if not isObjectDetected:
                    self.lidarSignals[ray] = PLAYER_SETTING.RADIUS_LIDAR
                    self.lidarVisualize[ray]["target"] = {
                        "x": target_x,
                        "y": target_y
                    }
                    self.lidarVisualize[ray]["color"] = COLOR.CYAN
                    startAngle += PLAYER_SETTING.STEP_ANGLE
                    continue

                distance = PLAYER_SETTING.RADIUS_LIDAR

                for obstacle in inRangeLidarObject:
                    scalar = (self.xPos-obstacle.xPos)*math.sin(startAngle) + \
                        (obstacle.yPos-self.yPos)*math.cos(startAngle)
                    if scalar > 0:
                        distance = min(distance, Utils.getDistanceFromObstacle(
                            self.xPos, self.yPos, target_x, target_y, obstacle.xPos, obstacle.yPos))

                if distance <= PLAYER_SETTING.RADIUS_LIDAR:
                    target_x = self.xPos - math.sin(startAngle) * distance
                    target_y = self.yPos + math.cos(startAngle) * distance
                    self.lidarSignals[ray] = distance
                    self.lidarVisualize[ray]["color"] = COLOR.RED
                    self.lidarVisualize[ray]["target"] = {
                        "x": target_x,
                        "y": target_y
                    }
                else:
                    self.lidarSignals[ray] = PLAYER_SETTING.RADIUS_LIDAR
                    self.lidarVisualize[ray]["color"] = COLOR.CYAN
                    self.lidarVisualize[ray]["target"] = {
                        "x": target_x,
                        "y": target_y
                    }
                # elif distance == INT_INFINITY:
                #     self.lidarVisualize[ray]["color"] = COLOR.CYAN
                # else:
                #     self.lidarVisualize[ray]["color"] = COLOR.PINK

                startAngle += PLAYER_SETTING.STEP_ANGLE

    def checkAchieveGoal(self):
        if self.yPos <= PLAYER_SETTING.Y_GOAL_POSITION:
            self.is_goal = True

    def draw(self, screen):
        # draw player on 2D board
        pygame.draw.circle(
            screen, COLOR.RED, (self.xPos, self.yPos), self.radiusObject
        )
        # draw player direction
        pygame.draw.line(screen, COLOR.GREEN, (self.xPos, self.yPos),
                         (self.xPos - math.sin(self.currAngle)*20,
                          self.yPos + math.cos(self.currAngle)*20), 3)

        for lidarItemVisualize in self.lidarVisualize:
            color = lidarItemVisualize["color"]
            srcX = lidarItemVisualize["source"]["x"]
            srcY = lidarItemVisualize["source"]["y"]
            targetX = lidarItemVisualize["target"]["x"]
            targetY = lidarItemVisualize["target"]["y"]
            pygame.draw.line(screen, color, (srcX, srcY), (targetX, targetY))

        if self.currRotationVelocity != 0:
            
            r = abs(self.currentForwardVelocity/self.currRotationVelocity)
            centerDireaction = 0
            if self.currRotationVelocity > 0:
                centerDireaction = self.currAngle + math.pi
            else:
                centerDireaction = self.currAngle

            centerX = self.xPos + r*math.cos(centerDireaction)
            centerY = self.yPos + r*math.sin(centerDireaction)

            pygame.draw.circle(screen, COLOR.WBLUE, (centerX,centerY), r,width=2)



class Obstacles(Car):
    def __init__(self, initX, initY):
        super().__init__(
            initX=initX,
            initY=initY,
            maxForwardVelocity=PLAYER_SETTING.MAX_FORWARD_VELO,
            minRotationVelocity=PLAYER_SETTING.MIN_ROTATION_VELO,
            maxRotationVelocity=PLAYER_SETTING.MAX_ROTATION_VELO,
            accelerationForward=PLAYER_SETTING.ACCELERATION_FORWARD,
            accelerationRotate=PLAYER_SETTING.ACCELERATION_ROTATE,
            radiusObject=OBSTACLE_SETTING.RADIUS_OBJECT,
        )
        self.currAngle = 0

    def draw(self, screen):
        # draw player on 2D board
        pygame.draw.circle(
            screen, COLOR.PINK, (self.xPos, self.yPos), self.radiusObject
        )

        # draw player direction
        pygame.draw.line(screen, COLOR.GREEN, (self.xPos, self.yPos),
                         (self.xPos - math.sin(self.currAngle)*20,
                          self.yPos + math.cos(self.currAngle)*20), 3)


class PyGame2D():
    def __init__(self):
        pygame.init()
        self.visualMap = np.zeros((GAME_SETTING.SCREEN_WIDTH,GAME_SETTING.SCREEN_HEIGHT,3),dtype='uint8')
        self.screen = pygame.display.set_mode(
            (GAME_SETTING.SCREEN_WIDTH, GAME_SETTING.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.robot = Robot()
        self.lane = Lane()
        self.obstacles = self._initObstacle()
        center, self.frame = genData(GAME_SETTING.SCREEN_WIDTH)
        self.obstacles[0].xPos = center[0]
        self.obstacles[0].yPos = center[1]
        self.preRobotYPos = self.robot.yPos
        self.mode = 0
        self.stepCounter = 0
        self.totalReward = 0
        self.preAngle = self.robot.currAngle
        self.tmp = []
        

    def _initObstacle(self):
        # TODO: UPDATE IT
        # numOfObstacles = random.randrange(OBSTACLE_SETTING.MAX_INSTANCES)
        numOfObstacles = OBSTACLE_SETTING.MAX_INSTANCES
        obstacles = []
        for _ in range(numOfObstacles):
            obstacleInstance = Obstacles(
                initX=GAME_SETTING.SCREEN_WIDTH//2 +
                random.randint(-int(0.1*GAME_SETTING.SCREEN_WIDTH//2),
                               int(0.1*GAME_SETTING.SCREEN_WIDTH//2)),
                initY=random.randint(0, int(0.7*GAME_SETTING.SCREEN_HEIGHT))
            )
            cv2.circle(self.visualMap,(obstacleInstance.xPos,obstacleInstance.yPos),PLAYER_SETTING.RADIUS_OBJECT,(0,0,255),-1)
            obstacles.append(obstacleInstance)
        return obstacles

    def _obstacleMoves(self):
        for obstacle in self.obstacles:
            keys = ACTIONS_LIST
            probs = OBSTACLE_SETTING.PROBABILITIES_ACTION
            # randomIndex = random.choices(range(len(keys)), probs)[0]
            randomIndex = 2
            choosedKey = keys[randomIndex]
            obstacle.move(action=choosedKey)

    def action(self, action):
        self.stepCounter += 1
        self.lastAction = action
        preX, preY = self.robot.xPos, self.robot.yPos
        self.robot.move(action=action)
        cv2.line(self.visualMap,(int(preX),int(preY)),(int(self.robot.xPos),int(self.robot.yPos)),(0,255,0),1)
        self._obstacleMoves()
        self.robot.checkCollision(collisions=self.obstacles, lane=self.lane)
        self.robot.scanLidar(obstacles=self.obstacles)
        self.robot.checkAchieveGoal()

    def is_done(self):
        if ((not self.robot.is_alive) or self.robot.is_goal):
            return True
        return False

    def view(self):
        # draw game
        self.screen.fill(COLOR.BLACK)
        self.screen.blit(self.screen, (0, 0))
        self.clock.tick(GAME_SETTING.FPS)
        self.lane.draw(screen=self.screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.robot.draw(screen=self.screen)
        for each in self.tmp:
            pygame.draw.circle(self.screen,COLOR.WBLUE,(each.xPos,each.yPos),20,2)
        # pygame.draw.circle(self.screen,COLOR.WBLUE,(self.robot.xPos + 10*math.cos(self.robot.currAngle + math.pi/2),self.robot.yPos + 10*math.sin(self.robot.currAngle + math.pi/2)),10,2 )
        self.tmp = []
        for obstacle in self.obstacles:
            obstacle.draw(screen=self.screen)
        pygame.display.flip()

    @staticmethod
    def controller(mode=GAME_SETTING.MODE_MANUALLY, game=None):
        if game == None:
            return
        if mode == GAME_SETTING.MODE_MANUALLY:
            while True:
                Utils.inputUser(game)
                game.view()
        elif mode == GAME_SETTING.MODE_AUTO:        
            dstar = atdstar.DStarService(start=(0, 0), goal=(1,1), map=Gen.genMap())    
            while True:
                Utils.inputUser(game)
                game.view()
                # Get the current (x,y) coordinate of robot
                robotX, robotY = game.robot.xPos, game.robot.yPos

                # Get the list signals lidar [(x, y), (x, y), (x, y), ...]
                listCoordinateLidarSignals = []
                lidarsSignal = game.robot.lidarSignals
                lidarsVisualization = game.robot.lidarVisualize
                for idx, distance in enumerate(lidarsSignal):
                    if distance <= INT_INFINITY:
                        targetX = lidarsVisualization[idx]["target"]["x"]
                        targetY = lidarsVisualization[idx]["target"]["y"]
                        if targetX > 0 and targetY > 0 and targetX < GAME_SETTING.SCREEN_WIDTH and targetY < GAME_SETTING.SCREEN_HEIGHT:
                            listCoordinateLidarSignals.append(
                                (int(targetX), int(targetY)))

                # Build the binary map with 0 is nothing, 1 is obstacle
                obstacleMap = [[D_STAR.ENV.NO_OBS for _ in range(
                    GAME_SETTING.SCREEN_WIDTH)] for _ in range(GAME_SETTING.SCREEN_HEIGHT)]
                obstacleMap = np.asarray(obstacleMap)
                
                
                widthImg = GAME_SETTING.SCREEN_WIDTH
                heighImg = GAME_SETTING.SCREEN_HEIGHT
                
                addedFakeObstacle = np.empty((0, 2), int)
                
                for coor in listCoordinateLidarSignals:
                    targetX, targetY = coor[0], coor[1]
                    obstacleMap[targetY][targetX] = D_STAR.ENV.HAS_OBS
                    for deltaY in range(-10, 10):
                        for deltaX in range(-10, 10):                                
                            m, n = targetX + deltaX, targetY + deltaY
                            if m >= 0 and m < widthImg and n >= 0 and n < heighImg and targetX is not m and targetY is not n:
                                addedFakeObstacle = np.append(addedFakeObstacle, [[m, n]], axis=0)
                                obstacleMap[n][m] = D_STAR.ENV.HAS_OBS
                # Convert to the transformation map
                # TODO: Do it

                # Put the transformation map into D* algorithm to get the path
                startPoint = (int(robotX), int(robotY))
                startPointTransform = ((int(robotX)), 0)                
                goalPoint = (GAME_SETTING.SCREEN_WIDTH//2, 5)
                goalPointTransform = (GAME_SETTING.SCREEN_WIDTH//2, 5)
                obstacleMapTransform = obstacleMap[startPoint[1] : startPoint[1] + PLAYER_SETTING.RADIUS_LIDAR, :]
                dstar.onReset(startPointTransform, goalPointTransform, obstacleMapTransform)
                
                pathToGoal = dstar.getPath()                
                pathToGoal = map(np.array, pathToGoal)
                pathToGoal = np.array(list(pathToGoal))    
                
                # print(addedFakeObstacle)
                coorObstacles = map(np.array, listCoordinateLidarSignals)  
                coorObstacles = np.array(list(coorObstacles))
                coorObstacles = np.append(coorObstacles, addedFakeObstacle, axis=0)

                # Visualization (optional)
                # Create OPENCV visualizationT
                img = np.zeros((heighImg, widthImg), dtype=np.uint8)
                if (len(pathToGoal)):
                    img[pathToGoal[:, 1], pathToGoal[:, 0]] = 255
                if (len(coorObstacles)):
                    img[coorObstacles[:, 1], coorObstacles[:, 0]] = 255
                                                            
                
                # img[transformIndexes] = 255
                cv2.circle(img, startPoint, 10, 255, 2)
                cv2.circle(img, goalPoint, 10, 255, 2)
                cv2.imshow("asa", img)                
                # Decide the action suitable to the map


game = PyGame2D()
game.controller(mode=GAME_SETTING.MODE_AUTO, game=game)
