import math
import pygame
from sys import exit
from const import *
from utils import *


class Lidar():
    def __init__(self) -> None:
        pass


class Car():
    def __init__(self, initX, initY, maxForwardVelocity, maxRotationVelocity, accelerationForward, accelerationRotate, radiusObject) -> None:
        self.xPos, self.yPos = initX, initY
        self.maxForwardVelocity = maxForwardVelocity
        self.maxRotationVelocity = maxRotationVelocity

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
                self.currRotationVelocity - self.accelerationRotate, PLAYER_SETTING.MIN_ROTATION_VELO)
        elif action == ACTIONS.TURN_RIGHT_ACCELERATION:
            self.currRotationVelocity = min(
                self.currRotationVelocity + self.accelerationRotate, PLAYER_SETTING.MAX_ROTATION_VELO)
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

    def checkCollision(self, collisions):
        if collisions == None or len(collisions):
            return False

        for collision in collisions:
            distanceBetweenCenter = Utils.distanceBetweenTwoPoints(
                self.xPos, self.yPos, collision.xPos, collision.yPos)
            # https://stackoverflow.com/questions/22135712/pygame-collision-detection-with-two-circles
            if distanceBetweenCenter <= 2*PLAYER_SETTING.RADIUS_OBJECT:
                return True

        return False

    def _rayCasting(self, collisions):
        pass

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
        self.robot = Car(
            initX=PLAYER_SETTING.INITIAL_X,
            initY=PLAYER_SETTING.INITIAL_Y,
            maxForwardVelocity=PLAYER_SETTING.MAX_FORWARD_VELO,
            maxRotationVelocity=PLAYER_SETTING.MAX_ROTATION_VELO,
            accelerationForward=PLAYER_SETTING.ACCELERATION_FORWARD,
            accelerationRotate=PLAYER_SETTING.ACCELERATION_ROTATE,
            radiusObject=PLAYER_SETTING.RADIUS_OBJECT,
        )
        self.obstacles = self._initObstacle()
        self.mode = 0

    def _initObstacle(self):
        pass

    def action(self, action):
        self.robot.move(action=action)
        self.robot.checkCollision(collisions=self.obstacles)

    def evaluate(self):
        # TODO: UPDATE IT
        reward = 0
        """
        if self.robot.check_flag:
            self.robot.check_flag = False
            reward = 2000 - self.robot.time_spent
            self.robot.time_spent = 0
        """
        # if not self.robot.is_alive:
        #     reward = -10000 + self.robot.distance

        # elif self.robot.goal:
        #     reward = 10000
        # return reward

    def is_done(self):
        # TODO: UPDATE IT
        # if not self.robot.is_alive or self.robot.goal:
        #     self.robot.current_check = 0
        #     self.robot.distance = 0
        #     return True
        # return False
        pass

    def observe(self):
        pass
        # TODO: UPDATE IT
        # radars = self.robot.radars
        # ret = [0, 0, 0, 0, 0]
        # for i, r in enumerate(radars):
        #     ret[i] = int(r[1] / 30)

        # return tuple(ret)

    def view(self):
        # TODO: UPDATE IT
        # draw game
        self.screen.fill(COLOR.BLACK)
        self.screen.blit(self.screen, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.robot.draw(screen=self.screen)
        pygame.display.flip()
        self.clock.tick(GAME_SETTING.FPS)


game = PyGame2D()
while True:
    Utils.inputUser(game)
    game.view()
    pass
