from collections import namedtuple
from enum import Enum
import pygame

# reset
# reward
# play (action) -> direction
# game_iteration
# is_collision

Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


BLOCK_SIZE = 20
SPEED = 20

class Action(Enum):
    INCREASE_VELOCITY = 1
    DECREASE_VELOCITY = 2
    INCREASE_ANGULAR_VELOCITY = 3
    DECREASE_ANGUALR_VELOCITY = 4

class CarGame:
    def __init__(self, width=640, height=480):
        pygame.init()
        self.width = width
        self.height = height
        
        # Step 1: Init display
        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Step 2: Init game state
        self.currentPosition = 0
        self.score = 0
        
    def play_step (self):
        print("play step")
        return False, 2
    
    def is_collision(self):
        print("Alo")
        
    def _move(self, action):
        print("nana")
        
if __name__ == "__main__":
    game = CarGame()
    while True:
        game_over, score = game.play_step()
        if game_over == True:
            break
    print('Final score', score)
    