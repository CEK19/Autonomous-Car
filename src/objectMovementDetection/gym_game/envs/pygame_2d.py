import math
import pygame
from sys import exit
import random
from const import *
from utils import *


class Lidar():
    def __init__(self) -> None:
        pass


class Car():
    def __init__(self, initX, initY, maxForwardVelocity, maxRotationVelocity, accelerationForward, accelerationRotate) -> None:
        self.xPos, self.yPos = initX, initY
        self.maxForwardVelocity = maxForwardVelocity
        self.maxRotationVelocity = maxRotationVelocity

        self.currentForwardVelocity = 0  # always >= 0
        self.currRotationVelocity = 0  # rotate left < 0, rotate right > 0

        self.currAngle = math.pi
        self.accelerationForward = accelerationForward
        self.accelerationRotate = accelerationRotate

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
        self.yPos += math.cos(self.currAngle) * self.currVelocity * dt
        self.xPos += -math.sin(self.currAngle) * self.currVelocity * dt

    def checkCollision(self, collisions):
        for collision in collisions:
            distanceBetweenCenter = Utils.distanceBetweenTwoPoints(
                self.xPos, self.yPos, collision.xPos, collision.yPos)
            # https://stackoverflow.com/questions/22135712/pygame-collision-detection-with-two-circles
            if distanceBetweenCenter <= 2*PLAYER_SETTING.RADIUS_OBJECT:
                return True
        return False

    def _rayCasting(self, collisions):
        pass


class PyGame2D():
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (GAME_SETTING.SCREEN_WIDTH, GAME_SETTING.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.robot = Car()
        self.obstacles = self._initObstacle()
        self.mode = 0

    def _initObstacle():
        pass

    def action(self, action):
        self.robot.move(action=action)

        # TODO: UPDATE IT
        self.car.update()
        self.car.checkCollision(collisions=self.obstacles)

        self.car.radars.clear()
        for d in range(-90, 120, 45):
            self.car.check_radar(d)

    def evaluate(self):
        # TODO: UPDATE IT
        reward = 0
        """
        if self.car.check_flag:
            self.car.check_flag = False
            reward = 2000 - self.car.time_spent
            self.car.time_spent = 0
        """
        if not self.car.is_alive:
            reward = -10000 + self.car.distance

        elif self.car.goal:
            reward = 10000
        return reward

    def is_done(self):
        # TODO: UPDATE IT
        if not self.car.is_alive or self.car.goal:
            self.car.current_check = 0
            self.car.distance = 0
            return True
        return False

    def observe(self):
        # TODO: UPDATE IT
        radars = self.car.radars
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)

        return tuple(ret)

    def view(self):
        # TODO: UPDATE IT
        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode += 1
                    self.mode = self.mode % 3

        self.screen.blit(self.car.map, (0, 0))

        if self.mode == 1:
            self.screen.fill((0, 0, 0))

        self.car.radars_for_draw.clear()
        for d in range(-90, 120, 45):
            self.car.check_radar_for_draw(d)

        pygame.draw.circle(self.screen, (255, 255, 0),
                           check_point[self.car.current_check], 70, 1)
        self.car.draw_collision(self.screen)
        self.car.draw_radar(self.screen)
        self.car.draw(self.screen)

        text = self.font.render(
            "Press 'm' to change view mode", True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 100)
        self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(self.game_speed)
