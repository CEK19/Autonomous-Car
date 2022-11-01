
import math
import pygame
from sys import exit
import random
from const import *
from utils import *
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
        self.raycastingLists = []

    def _move(self):
        dt = float(1/GameSettingParam.FPS)

        self.yPos += math.cos(self.currAngle) * self.currVelocity * dt
        self.xPos += -math.sin(self.currAngle) * self.currVelocity * dt
        self.currAngle += self.currRotationVelocity*dt

    def _playerInput(self):
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

    def _rayCasting(self):
        global obstacles
        startAngle = self.currAngle - PlayerParam.HALF_FOV
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

                for obstacle in obstacles:
                    distance = Utils.distanceBetweenTwoPoints(
                        target_x, target_y, obstacle.xPos, obstacle.yPos)
                    if distance <= PlayerParam.RADIUS_OBJECT:
                        isDetectObject = True
                        pygame.draw.line(GLOBAL_SCREEN, CustomColor.WHITE,
                                         (self.xPos, self.yPos), (target_x, target_y))
                        break
                    if depth == PlayerParam.RADIUS_LIDAR and not isDetectObject:
                        pygame.draw.line(GLOBAL_SCREEN, CustomColor.WHITE,
                                         (self.xPos, self.yPos), (target_x, target_y))

            startAngle += PlayerParam.STEP_ANGLE

    def _checkCollision(self):
        global obstacles
        for obstacle in obstacles:
            distanceBetweenCenter = Utils.distanceBetweenTwoPoints(
                self.xPos, self.yPos, obstacle.xPos, obstacle.yPos)
            # https://stackoverflow.com/questions/22135712/pygame-collision-detection-with-two-circles
            if distanceBetweenCenter <= 2*PlayerParam.RADIUS_OBJECT:
                print(datetime.datetime.now(), "ouchhhh")
                pass

    def draw(self):
        global GLOBAL_SCREEN
        self._playerInput()
        self._rayCasting()
        self._checkCollision()
        self._move()

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
        keys = [PlayerParam.INC_ROTATION_VELO,
                PlayerParam.DESC_ROTATION_VELO,
                PlayerParam.STOP,
                PlayerParam.INC_FORWARD_VELO,
                PlayerParam.DESC_FORWARD_VELO]
        probs = [
            0.1,
            0.1,
            0.1,
            0.4,
            0.3
        ]

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

        # draw player on 2D board
        pygame.draw.circle(GLOBAL_SCREEN, CustomColor.GREEN,
                           (self.xPos, self.yPos), PlayerParam.RADIUS_OBJECT)

        pygame.draw.circle(GLOBAL_SCREEN, CustomColor.RED,
                           (self.xPos, self.yPos), 6)
        # draw player direction
        pygame.draw.line(GLOBAL_SCREEN, CustomColor.GREEN, (self.xPos, self.yPos),
                         (self.xPos - math.sin(self.currAngle) * 20,
                          self.yPos + math.cos(self.currAngle) * 20), 3)
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

# Start game
while True:
    GLOBAL_CLOCK.tick(GameSettingParam.FPS)
    GLOBAL_SCREEN.fill(CustomColor.BLACK)
    GLOBAL_SCREEN.blit(GLOBAL_SCREEN, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    player.draw()
    for obstacle in obstacles:
        obstacle.draw()

    pygame.display.flip()
