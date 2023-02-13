import math
import pygame
from sys import exit
from const import *
from utils import *
import random
import numpy as np


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
            initX=PLAYER_SETTING.INITIAL_X,
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
        self.lidarSignals = [PLAYER_SETTING.RADIUS_LIDAR]*PLAYER_SETTING.CASTED_RAYS
        self.lidarVisualize = [{"source": {"x": self.xPos, "y": self.yPos},
                                "target": {"x": self.xPos, "y": self.yPos},
                                "color": COLOR.WHITE
                                } for x in range(PLAYER_SETTING.CASTED_RAYS)]

    def checkCollision(self, collisions):
        if collisions == None or len(collisions) == 0:
            self.is_alive = True

        for collision in collisions:
            distanceBetweenCenter = Utils.distanceBetweenTwoPoints(
                self.xPos, self.yPos, collision.xPos, collision.yPos)
            if distanceBetweenCenter <= PLAYER_SETTING.RADIUS_OBJECT + OBSTACLE_SETTING.RADIUS_OBJECT:
                self.is_alive = False
                return

        if self.xPos <= 0 or self.xPos >= GAME_SETTING.SCREEN_WIDTH or self.yPos < 0 or self.yPos >= GAME_SETTING.SCREEN_HEIGHT:
            self.is_alive = False

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
    def __init__(self, initX, initY) -> None:
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
            screen, COLOR.RED, (self.xPos, self.yPos), self.radiusObject
        )

        # draw player direction
        pygame.draw.line(screen, COLOR.GREEN, (self.xPos, self.yPos),
                         (self.xPos - math.sin(self.currAngle)*20,
                          self.yPos + math.cos(self.currAngle)*20), 3)


class PyGame2D():
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (GAME_SETTING.SCREEN_WIDTH, GAME_SETTING.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.robot = Robot()
        self.obstacles = self._initObstacle()
        self.preRobotYPos = self.robot.yPos
        self.mode = 0
        self.stepCounter = 0
        self.totalReward = 0
        self.tmp = []

    def _initObstacle(self):
        #TODO: UPDATE IT
        # numOfObstacles = random.randrange(OBSTACLE_SETTING.MAX_INSTANCES)
        numOfObstacles = OBSTACLE_SETTING.MAX_INSTANCES
        obstacles = []
        for _ in range(numOfObstacles):
            obstacleInstance = Obstacles(
                initX=GAME_SETTING.SCREEN_WIDTH//2 +
                random.randint(-int(0.8*GAME_SETTING.SCREEN_WIDTH//2),
                               int(0.8*GAME_SETTING.SCREEN_WIDTH//2)),
                initY=random.randint(0, int(0.7*GAME_SETTING.SCREEN_HEIGHT))
            )
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
        self.robot.move(action=action)
        self._obstacleMoves()
        self.robot.checkCollision(collisions=self.obstacles)
        self.robot.scanLidar(obstacles=self.obstacles)
        self.robot.checkAchieveGoal()

    def evaluate(self):
        # tmpLidarSignal = self.robot.lidarSignals

        # directionVector = [math.cos(self.robot.currAngle),math.sin(self.robot.currAngle)]
        # objInRange = []

        reward = 0

        minDistance = INT_INFINITY

        ahead = [self.robot.xPos + 10*math.cos(self.robot.currAngle + math.pi/2),self.robot.yPos + 10*math.sin(self.robot.currAngle + math.pi/2)]
        
        if self.robot.currRotationVelocity != 0:
            
            r = abs(self.robot.currentForwardVelocity/self.robot.currRotationVelocity)
            centerDireaction = 0
            if self.robot.currRotationVelocity > 0:
                centerDireaction = self.robot.currAngle + math.pi
            else:
                centerDireaction = self.robot.currAngle

            centerX = self.robot.xPos + r*math.cos(centerDireaction)
            centerY = self.robot.yPos + r*math.sin(centerDireaction)

            for eachObj in self.obstacles:
                d = abs(r - Utils.distanceBetweenTwoPoints(eachObj.xPos,eachObj.yPos,centerX,centerY))
                if d < PLAYER_SETTING.RADIUS_OBJECT*4:
                    distanceToRobot = Utils.distanceBetweenTwoPoints(eachObj.xPos,eachObj.yPos,self.robot.xPos,self.robot.yPos)
                    if Utils.distanceBetweenTwoPoints(ahead[0],ahead[1],eachObj.xPos,eachObj.yPos) < distanceToRobot and \
                        distanceToRobot < PLAYER_SETTING.RADIUS_LIDAR:
                        # self.tmp.append(eachObj)
                        minDistance = min(minDistance,d)
        else:
            for eachObj in self.obstacles:
                s2 = Utils.distanceBetweenTwoPoints(eachObj.xPos,eachObj.yPos,self.robot.xPos,self.robot.yPos)
                s1 = Utils.distanceBetweenTwoPoints(ahead[0],ahead[1],eachObj.xPos,eachObj.yPos)

                d = (s2**2-s1**2-100)/20
                h = math.sqrt(abs(s1**2-d**2))

                if  h < PLAYER_SETTING.RADIUS_OBJECT*4 and s2 > s1 and s2 < PLAYER_SETTING.RADIUS_LIDAR:
                    # self.tmp.append(eachObj)
                    minDistance = min(minDistance,h)
        

        if minDistance == INT_INFINITY:
            reward += 100
        else:
            reward += -100 + 200*minDistance/(PLAYER_SETTING.RADIUS_OBJECT*4)
            # print(minDistance/(PLAYER_SETTING.RADIUS_OBJECT*4))

        # s1 = reward
        
        if not self.robot.is_alive:
            reward += -10000

        if (GAME_SETTING.SCREEN_WIDTH < self.robot.yPos):
            reward += -100000000

        ratioLeft = (self.robot.xPos)/(GAME_SETTING.SCREEN_WIDTH)

        reward += 200 - abs(ratioLeft-0.5)*800

        reward += (self.preRobotYPos - self.robot.yPos)*100
        self.preRobotYPos = self.robot.yPos

        reward += -abs(self.robot.currAngle - math.pi)*10

        reward += -self.stepCounter*0.5

        

        if self.lastAction == ACTIONS.FORWARD_ACCELERATION:
            reward += 50
        elif self.lastAction == ACTIONS.BACKWARD_ACCELERATION:
            reward -= 10
        elif self.lastAction == ACTIONS.TURN_LEFT_ACCELERATION:
            reward -= 5
        elif self.lastAction == ACTIONS.TURN_RIGHT_ACCELERATION:
            reward -= 5

        if self.robot.is_goal:
            reward += (PLAYER_SETTING.MAX_STEP_PER_EPOCH - self.stepCounter)*(self.totalReward/self.stepCounter)

        self.totalReward += reward
        return reward
        # return [int(s1),int(s2),int(s3),int(0),int(0),ratioLeft,reward]

    def is_done(self):
        if ((not self.robot.is_alive) or self.robot.is_goal or self.stepCounter > PLAYER_SETTING.MAX_STEP_PER_EPOCH):
            return True
        return False

    def observe(self):
        ratioLeft = (self.robot.xPos)/(GAME_SETTING.SCREEN_WIDTH)
        alpha = self.robot.currAngle
        fwVelo = self.robot.currentForwardVelocity
        rVelo = self.robot.currRotationVelocity
        lidars = self.robot.lidarSignals        
        
        infoStateVector = np.array([ratioLeft, alpha, fwVelo, rVelo])
        lidarStateVector = np.array(lidars)        
        return np.concatenate((infoStateVector, lidarStateVector))

    def view(self):
        # draw game
        self.screen.fill(COLOR.BLACK)
        self.screen.blit(self.screen, (0, 0))

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
        self.clock.tick(GAME_SETTING.FPS)

# import time

# game = PyGame2D()
# ct = 0
# while True:
#     Utils.inputUser(game)
#     ct+= 1
#     game.view()
#     data = game.evaluate()
#     # print(f"{data[0]:>15} {data[1]:>15} {data[2]:>15} {data[3]:>15} {data[4]:>15} {data[5]:>15} {data[6]:>20}")
#     print(int(data),"      ",ct)
#     pass
