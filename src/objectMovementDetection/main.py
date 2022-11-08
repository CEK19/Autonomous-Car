
import math
import pygame
from sys import exit
import random
from const import *
from utils import *
from table import *
import datetime


class Player():
    def __init__(self, maxVelocity, maxRotationVelocity):
        super().__init__()
        global GLOBAL_SCREEN
        self.xPos, self.yPos = PlayerParam.INITIAL_X, PlayerParam.INITIAL_Y
        self.maxVelocity = maxVelocity
        self.maxRotationVelocity = maxRotationVelocity

        self.currVelocity = 0  # always >= 0
        self.currRotationVelocity = 0  # rotate left < 0, rotate right > 0
        self.currAngle = math.pi
        self.accelerationForward = PlayerParam.ACCELERATION_FORWARD

        self.circleRect = pygame.draw.circle(
            GLOBAL_SCREEN, CustomColor.RED, (self.xPos, self.yPos), PlayerParam.RADIUS_OBJECT)

        # Raycasting
        self.rayCastingLists = [PlayerParam.INFINITY] * PlayerParam.CASTED_RAYS

        self.mode = MODE_PLAY.MANUAL
        self.displayGUI = GUI.DISPLAY

    def _move(self):
        dt = float(1/GameSettingParam.FPS)

        self.yPos += math.cos(self.currAngle) * self.currVelocity * dt
        self.xPos += -math.sin(self.currAngle) * self.currVelocity * dt
        self.currAngle += self.currRotationVelocity*dt

    def _playerInput(self, actionIndex):
        if (self.mode == MODE_PLAY.MANUAL):
            keys = pygame.key.get_pressed()

            # Rotate left ()
            if keys[pygame.K_a]:
                self.currRotationVelocity -= PlayerParam.ACCELERATION_ROTATE
            # Rotate right ()
            if keys[pygame.K_d]:
                self.currRotationVelocity += PlayerParam.ACCELERATION_ROTATE

            # Increase forward velocity
            if keys[pygame.K_w]:
                self.currVelocity = min(
                    self.currVelocity + PlayerParam.ACCELERATION_FORWARD, self.maxVelocity)

            # Stop
            if keys[pygame.K_s]:
                self.currVelocity = 0
                self.currRotationVelocity = 0

            # Decrease forward velocity
            if keys[pygame.K_x]:
                self.currVelocity = max(
                    self.currVelocity - PlayerParam.ACCELERATION_FORWARD, 0)

        elif (self.mode == MODE_PLAY.RL_TRAIN):

            if RLParam.ACTIONS[actionIndex] == PlayerParam.DESC_ROTATION_VELO:
                self.currRotationVelocity -= PlayerParam.ACCELERATION_ROTATE

            if RLParam.ACTIONS[actionIndex] == PlayerParam.INC_ROTATION_VELO:
                self.currRotationVelocity += PlayerParam.ACCELERATION_ROTATE

            if RLParam.ACTIONS[actionIndex] == PlayerParam.STOP:
                self.currVelocity = 0
                self.currRotationVelocity = 0

            if RLParam.ACTIONS[actionIndex] == PlayerParam.INC_FORWARD_VELO:
                self.currVelocity = min(
                    self.currVelocity + PlayerParam.ACCELERATION_FORWARD, self.maxVelocity)

            if RLParam.ACTIONS[actionIndex] == PlayerParam.DESC_FORWARD_VELO:
                self.currVelocity = max(
                    self.currVelocity - PlayerParam.ACCELERATION_FORWARD, 0)

    def _rayCasting(self):
        global obstacles
        inRangedObj = []

        for obstacle in obstacles:
            if Utils.distanceBetweenTwoPoints(self.xPos, self.yPos, obstacle.xPos, obstacle.yPos) < PlayerParam.RADIUS_OBJECT*2 + PlayerParam.RADIUS_LIDAR:
                inRangedObj.append(obstacle)
        startAngle = self.currAngle - PlayerParam.HALF_FOV
        if len(inRangedObj) == 0:
            for ray in range(PlayerParam.CASTED_RAYS):
                target_x = self.xPos - math.sin(startAngle) * PlayerParam.RADIUS_LIDAR
                target_y = self.yPos + math.cos(startAngle) * PlayerParam.RADIUS_LIDAR
                pygame.draw.line(GLOBAL_SCREEN, CustomColor.WHITE, (self.xPos, self.yPos), (target_x, target_y))
                self.rayCastingLists[ray] = PlayerParam.INFINITY
                startAngle += PlayerParam.STEP_ANGLE
        else:
            for ray in range(PlayerParam.CASTED_RAYS):
                # get ray target coordinates
                isDetectObject = False

                for depth in range(PlayerParam.RADIUS_LIDAR + 1):
                    if (isDetectObject):
                        break
                    target_x = self.xPos - \
                        math.sin(startAngle) * depth
                    target_y = self.yPos + \
                        math.cos(startAngle) * depth

                    for obstacle in inRangedObj:
                        distance = Utils.distanceBetweenTwoPoints(target_x, target_y, obstacle.xPos, obstacle.yPos)
                        if distance <= PlayerParam.RADIUS_OBJECT:
                            self.rayCastingLists[ray] = Utils.distanceBetweenTwoPoints(target_x, target_y, self.xPos, self.yPos)
                            isDetectObject = True
                            if self.displayGUI == GUI.DISPLAY:
                                pygame.draw.line(GLOBAL_SCREEN, CustomColor.CYAN, (self.xPos, self.yPos), (target_x, target_y))
                            break
                        if depth == PlayerParam.RADIUS_LIDAR and not isDetectObject:
                            self.rayCastingLists[ray] = PlayerParam.INFINITY
                            if self.displayGUI == GUI.DISPLAY:
                                pygame.draw.line(GLOBAL_SCREEN, CustomColor.WHITE, (self.xPos, self.yPos), (target_x, target_y))

                startAngle += PlayerParam.STEP_ANGLE

    def _checkCollision(self):
        global obstacles
        for obstacle in obstacles:
            distanceBetweenCenter = Utils.distanceBetweenTwoPoints(
                self.xPos, self.yPos, obstacle.xPos, obstacle.yPos)
            # https://stackoverflow.com/questions/22135712/pygame-collision-detection-with-two-circles
            if distanceBetweenCenter <= 2*PlayerParam.RADIUS_OBJECT:
                # print("Ouch!!!")
                pass

    def draw(self, actionIndex):
        global GLOBAL_SCREEN
        self._playerInput(actionIndex=actionIndex)
        self._rayCasting()
        self._checkCollision()
        self._move()

        if (self.displayGUI == GUI.DISPLAY):
            # draw player on 2D board
            pygame.draw.circle(GLOBAL_SCREEN, CustomColor.RED,
                               (self.xPos, self.yPos), PlayerParam.RADIUS_OBJECT)

            # draw player direction
            pygame.draw.line(GLOBAL_SCREEN, CustomColor.GREEN, (self.xPos, self.yPos),
                             (self.xPos - math.sin(self.currAngle) * 20,
                              self.yPos + math.cos(self.currAngle) * 20), 3)


class Obstacle(Player):
    def __init__(self):
        super().__init__(maxVelocity=PlayerParam.MAX_VELOCITY,
                         maxRotationVelocity=PlayerParam.MAX_ROTATION_VELOCITY)

        self.xPos = ObstacleParam.INITIAL_OBSTACLE_X + random.randint(-int(0.8*ObstacleParam.INITIAL_OBSTACLE_X), int(
            0.8*ObstacleParam.INITIAL_OBSTACLE_X))

        self.yPos = ObstacleParam.INITIAL_OBSTACLE_Y + random.randint(0, int(
            0.7*GameSettingParam.HEIGHT))

        self.circleRect = pygame.draw.circle(
            GLOBAL_SCREEN, CustomColor.GREEN, (self.xPos, self.yPos), PlayerParam.RADIUS_OBJECT)
        self.currAngle = 0

        # Is random ?
        self.randomVelo = False

    def _playerInput(self):
        keys = RLParam.ACTIONS
        probs = ObstacleParam.PROBABILITIES_ACTION

        # choosedKey = keys[random.choice(range(len(keys)))]
        randomIndex = random.choices(range(len(keys)), probs)[0]
        choosedKey = keys[randomIndex]

        if choosedKey == PlayerParam.INC_ROTATION_VELO:
            self.currRotationVelocity += ObstacleParam.OBSTACLE_ACCELERATION_ROTATE
        if choosedKey == PlayerParam.DESC_ROTATION_VELO:
            self.currRotationVelocity -= ObstacleParam.OBSTACLE_ACCELERATION_ROTATE
        if choosedKey == PlayerParam.STOP:
            self.currVelocity = 0
            self.currRotationVelocity = 0
        if choosedKey == PlayerParam.INC_FORWARD_VELO:
            self.currVelocity = min(
                self.currVelocity + ObstacleParam.OBSTACLE_ACCELERATION_FORWARD, self.maxVelocity)
        if choosedKey == PlayerParam.DESC_FORWARD_VELO:
            self.currVelocity = max(
                self.currVelocity - ObstacleParam.OBSTACLE_ACCELERATION_FORWARD, 0)

    def _rayCasting(self):
        pass

    def _checkCollision(self):
        pass

    def draw(self):
        global GLOBAL_SCREEN
        self._playerInput()
        self._move()

        if (self.displayGUI == GUI.DISPLAY):
            # draw player on 2D board
            pygame.draw.circle(GLOBAL_SCREEN, CustomColor.GREEN,
                               (self.xPos, self.yPos), PlayerParam.RADIUS_OBJECT)

            pygame.draw.circle(GLOBAL_SCREEN, CustomColor.RED,
                               (self.xPos, self.yPos), 6)
            # draw player direction
            pygame.draw.line(GLOBAL_SCREEN, CustomColor.GREEN, (self.xPos, self.yPos),
                             (self.xPos - math.sin(self.currAngle) * 20,
                              self.yPos + math.cos(self.currAngle) * 20), 3)


class Environment:
    def __init__(self, currentPlayer, currentObstacles):
        self.currPlayer = currentPlayer
        self.currObstacles = currentObstacles

        self.rayCastingData = currentPlayer.rayCastingLists
        self.xPos, self.yPos = currentPlayer.xPos, currentPlayer.yPos

        currentPlayer.mode = MODE_PLAY.RL_TRAIN
        currentPlayer.displayGUI = GUI.DISPLAY

        for obstacle in currentObstacles:
            obstacle.mode = MODE_PLAY.RL_TRAIN
            obstacle.displayGUI = GUI.DISPLAY
            
    def _isDoneEpisode(self):
        return self.yPos <= 0 or self.yPos > GameSettingParam.HEIGHT   or self.xPos <= 0 or self.xPos >= GameSettingParam.WIDTH

    def _selfUpdated(self):
        self.rayCastingData = self.currPlayer.rayCastingLists
        self.xPos, self.yPos = self.currPlayer.xPos, self.currPlayer.yPos

    def updateStateByAction(self, actionIndex):
        for obstacle in obstacles:
            obstacle.draw()
            
        self.currPlayer.draw(actionIndex=actionIndex)                    
        self._selfUpdated()
        
        nextState = RLAlgorithm.hashFromDistanceToState(
            signalPerAreaData=RLAlgorithm.convertRayCastingDataToSignalPerArea(rayCastingData=self.rayCastingData), 
            leftSideDistance=abs(self.xPos), 
            rightSideDistance=abs(self.xPos - GameSettingParam.WIDTH))
        
        reward = RLAlgorithm.getReward(
            currState=nextState, currActionIndex=actionIndex)
        
        done = self._isDoneEpisode()
        
        return nextState, reward, done
    
    def getCurrentState(self):
        return RLAlgorithm.hashFromDistanceToState(signalPerAreaData=RLAlgorithm.convertRayCastingDataToSignalPerArea(rayCastingData=self.rayCastingData),
                                                   leftSideDistance=abs(self.xPos),
                                                   rightSideDistance=abs(self.xPos - GameSettingParam.WIDTH))

    def reset(self):
        del self
        global player, obstacles
        player = Player(maxVelocity=PlayerParam.MAX_VELOCITY,
                maxRotationVelocity=PlayerParam.MAX_ROTATION_VELOCITY) 
        obstacles = []
        for _ in range(ObstacleParam.NUMBER_OF_OBSTACLES):
            obstacles.append(Obstacle())
        return Environment(currentPlayer=player, currentObstacles=obstacles)
                    
###########################################################################################

# Game setting
pygame.init()
GLOBAL_SCREEN = pygame.display.set_mode(
    (GameSettingParam.WIDTH, GameSettingParam.HEIGHT))
pygame.display.set_caption(GameSettingParam.CAPTION)
GLOBAL_CLOCK = pygame.time.Clock()

# Groups
player = Player(maxVelocity=PlayerParam.MAX_VELOCITY,
                maxRotationVelocity=PlayerParam.MAX_ROTATION_VELOCITY)
obstacles = []
for _ in range(ObstacleParam.NUMBER_OF_OBSTACLES):
    obstacles.append(Obstacle())


def startGame(mode=MODE_PLAY.MANUAL):
    if (mode == MODE_PLAY.MANUAL):
        while True:
            GLOBAL_CLOCK.tick(GameSettingParam.FPS)
            GLOBAL_SCREEN.fill(CustomColor.BLACK)
            GLOBAL_SCREEN.blit(GLOBAL_SCREEN, (0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            player.draw(actionIndex=None)            
            for obstacle in obstacles:
                obstacle.draw()

            pygame.display.flip()
    elif (mode == MODE_PLAY.RL_TRAIN):        
        env = Environment(currentPlayer=player, currentObstacles=obstacles)
        RL = RLAlgorithm(rayCastingData=env.rayCastingData,
                         actions=RLParam.ACTIONS)
        RL.train(env)


# startGame(mode=MODE_PLAY.RL_TRAIN)
startGame(mode=MODE_PLAY.MANUAL)
