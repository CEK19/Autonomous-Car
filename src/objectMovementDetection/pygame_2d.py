import math
import pygame
from sys import exit
from const import *
from utils import *
import random
import numpy as np
from datetime import datetime


ObstacleTest = pygame.Rect(300, 600, 500, 1000)


class Lane():
    def __init__(self) -> None:
        #  (left, top, width, height)
        self.outSideMostLeft = pygame.Rect(LANE_SETTING.OUTSIDE_LEFT_PADDING,
                                           LANE_SETTING.OUTSIDE_TOP_PADDING, LANE_SETTING.WIDTH_OF_LANE_BORDER,
                                           GAME_SETTING.SCREEN_HEIGHT -
                                           (LANE_SETTING.OUTSIDE_TOP_PADDING + LANE_SETTING.OUTSIDE_BOTTOM_PADDING))
        self.outSideMostTop = pygame.Rect(LANE_SETTING.OUTSIDE_LEFT_PADDING,
                                          LANE_SETTING.OUTSIDE_TOP_PADDING,
                                          GAME_SETTING.SCREEN_WIDTH -
                                          (LANE_SETTING.OUTSIDE_LEFT_PADDING +
                                           LANE_SETTING.OUTSIDE_RIGHT_PADDING),
                                          LANE_SETTING.WIDTH_OF_LANE_BORDER)
        self.outSideMostRight = pygame.Rect(GAME_SETTING.SCREEN_WIDTH - LANE_SETTING.OUTSIDE_RIGHT_PADDING,
                                            LANE_SETTING.OUTSIDE_TOP_PADDING,
                                            LANE_SETTING.WIDTH_OF_LANE_BORDER,
                                            GAME_SETTING.SCREEN_HEIGHT -
                                            (LANE_SETTING.OUTSIDE_TOP_PADDING + LANE_SETTING.OUTSIDE_BOTTOM_PADDING))
        self.outSideMostBottom = pygame.Rect(LANE_SETTING.OUTSIDE_LEFT_PADDING,
                                             GAME_SETTING.SCREEN_HEIGHT - LANE_SETTING.OUTSIDE_BOTTOM_PADDING,
                                             GAME_SETTING.SCREEN_WIDTH -
                                             (LANE_SETTING.OUTSIDE_LEFT_PADDING +
                                              LANE_SETTING.OUTSIDE_RIGHT_PADDING),
                                             LANE_SETTING.WIDTH_OF_LANE_BORDER)

        self.inSideMostLeft = pygame.Rect(LANE_SETTING.INSIDE_LEFT_PADDING,
                                          LANE_SETTING.INSIDE_TOP_PADDING, LANE_SETTING.WIDTH_OF_LANE_BORDER,
                                          GAME_SETTING.SCREEN_HEIGHT -
                                          (LANE_SETTING.INSIDE_TOP_PADDING + LANE_SETTING.INSIDE_BOTTOM_PADDING))
        self.inSideMostTop = pygame.Rect(LANE_SETTING.INSIDE_LEFT_PADDING,
                                         LANE_SETTING.INSIDE_TOP_PADDING,
                                         GAME_SETTING.SCREEN_WIDTH -
                                         (LANE_SETTING.INSIDE_LEFT_PADDING +
                                          LANE_SETTING.INSIDE_RIGHT_PADDING),
                                         LANE_SETTING.WIDTH_OF_LANE_BORDER)
        self.inSideMostRight = pygame.Rect(GAME_SETTING.SCREEN_WIDTH - LANE_SETTING.INSIDE_RIGHT_PADDING,
                                           LANE_SETTING.INSIDE_TOP_PADDING,
                                           LANE_SETTING.WIDTH_OF_LANE_BORDER,
                                           GAME_SETTING.SCREEN_HEIGHT -
                                           (LANE_SETTING.INSIDE_TOP_PADDING + LANE_SETTING.INSIDE_BOTTOM_PADDING))
        self.inSideMostBottom = pygame.Rect(LANE_SETTING.INSIDE_LEFT_PADDING,
                                            GAME_SETTING.SCREEN_HEIGHT - LANE_SETTING.INSIDE_BOTTOM_PADDING,
                                            GAME_SETTING.SCREEN_WIDTH -
                                            (LANE_SETTING.INSIDE_LEFT_PADDING +
                                             LANE_SETTING.INSIDE_RIGHT_PADDING),
                                            LANE_SETTING.WIDTH_OF_LANE_BORDER)

    def draw(self, screen):
        pygame.draw.rect(screen, COLOR.WHITE, self.outSideMostLeft)
        pygame.draw.rect(screen, COLOR.WHITE, self.outSideMostTop)
        pygame.draw.rect(screen, COLOR.WHITE, self.outSideMostRight)
        pygame.draw.rect(screen, COLOR.WHITE, self.outSideMostBottom)

        pygame.draw.rect(screen, COLOR.WHITE, self.inSideMostLeft)
        pygame.draw.rect(screen, COLOR.WHITE, self.inSideMostTop)
        pygame.draw.rect(screen, COLOR.WHITE, self.inSideMostRight)
        pygame.draw.rect(screen, COLOR.WHITE, self.inSideMostBottom)

    def isCollideWithThing(self, thing):
        return \
            self.outSideMostLeft.colliderect(thing) or self.outSideMostTop.colliderect(thing) or \
            self.outSideMostRight.colliderect(thing) or self.outSideMostBottom.colliderect(thing) or \
            self.inSideMostLeft.colliderect(thing) or self.inSideMostTop.colliderect(thing) or \
            self.inSideMostRight.colliderect(
                thing) or self.inSideMostBottom.colliderect(thing)


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
        self.lidarSignals = [INT_INFINITY]*PLAYER_SETTING.CASTED_RAYS
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
            # https://stackoverflow.com/questions/22135712/pygame-collision-detection-with-two-circles
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
                self.lidarSignals[ray] = INT_INFINITY
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
                    self.lidarSignals[ray] = INT_INFINITY
                    self.lidarVisualize[ray]["target"] = {
                        "x": target_x,
                        "y": target_y
                    }
                    self.lidarVisualize[ray]["color"] = COLOR.CYAN
                    startAngle += PLAYER_SETTING.STEP_ANGLE
                    continue

                distance = INT_INFINITY

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
                    self.lidarSignals[ray] = INT_INFINITY
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
        self.lane = Lane()
        self.obstacles = self._initObstacle()
        self.mode = 0

    def _initObstacle(self):
        # TODO: UPDATE IT
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
            randomIndex = random.choices(range(len(keys)), probs)[0]
            choosedKey = keys[randomIndex]
            obstacle.move(action=choosedKey)

    def action(self, action):
        self.robot.move(action=action)
        self._obstacleMoves()
        self.robot.checkCollision(collisions=self.obstacles, lane=self.lane)
        self.robot.scanLidar(obstacles=self.obstacles)
        self.robot.checkAchieveGoal()

    def evaluate(self):
        reward = 0
        if not self.robot.is_alive:
            reward += -1000

        if (GAME_SETTING.SCREEN_HEIGHT - self.robot.yPos) >= 1:
            reward += -10000

        frontLidars = self.robot.lidarSignals[75:105]
        eachFrontPenalty = 0.01

        leftLidars = self.robot.lidarSignals[0:75]
        eachLeftPenalty = 0.005

        rightLidars = self.robot.lidarSignals[105:180]
        eachRightPenalty = 0.005

        for frontLidar in frontLidars:
            if frontLidar == INT_INFINITY:
                reward += 1
                break
            reward += -abs(PLAYER_SETTING.RADIUS_LIDAR -
                           frontLidar)*eachFrontPenalty

        for leftLidar in leftLidars:
            if leftLidar == INT_INFINITY:
                reward += 1
                break
            reward += -abs(PLAYER_SETTING.RADIUS_LIDAR -
                           leftLidar)*eachLeftPenalty

        for rightLidar in rightLidars:
            if rightLidar == INT_INFINITY:
                reward += 1
                break
            reward += -abs(PLAYER_SETTING.RADIUS_LIDAR -
                           rightLidar)*eachRightPenalty

        ratioLeft = (self.robot.xPos)/(GAME_SETTING.SCREEN_WIDTH)

        if (ratioLeft <= 0.65 and ratioLeft >= 0.45):
            reward += ratioLeft*0.1
        else:
            reward += -ratioLeft*0.15

        return reward

    def is_done(self):
        if ((not self.robot.is_alive) or self.robot.is_goal):
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
        self.clock.tick(GAME_SETTING.FPS)
        self.lane.draw(screen=self.screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.robot.draw(screen=self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(screen=self.screen)
        pygame.display.flip()


game = PyGame2D()
while True:
    Utils.inputUser(game)
    game.view()
    pass
