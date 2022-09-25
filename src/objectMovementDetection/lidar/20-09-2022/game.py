import pygame
import time
import math
from utils import scale_image, blit_rotate_center, blit_text_center
import const
pygame.font.init()


RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)
WIDTH, HEIGHT = 500, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")
MAIN_FONT = pygame.font.SysFont(const.fontInfo["font"],const.fontInfo["size"])
FPS = const.config["FPS"]
LEFT_LINE = (200, 0, 5, HEIGHT)
RIGHT_LINE = (WIDTH - 200, 0, 5, HEIGHT)
class GameInfo:
    LEVELS = 10

    def __init__(self, level=1):
        self.level = level
        self.started = False
        self.level_start_time = 0

    def next_level(self):
        self.level += 1
        self.started = False

    def reset(self):
        self.level = 1
        self.started = False
        self.level_start_time = 0

    def game_finished(self):
        return self.level > self.LEVELS

    def start_level(self):
        self.started = True
        self.level_start_time = time.time()

    def get_level_time(self):
        if not self.started:
            return 0
        return round(time.time() - self.level_start_time)


class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0 # always >= 0
        self.rotation_vel = rotation_vel # rotate left < 0, rotate right > 0
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = const.player["acceleration"]

    def rotate(self, left=False, right=False):
        if left:
            # self.angle += self.rotation_vel
            self.rotation_vel -= const.player["acceleration_rotate"]
            print(self.rotation_vel)            
        elif right:
            self.rotation_vel += const.player["acceleration_rotate"]                    
            print(self.rotation_vel)            
        
    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)

    def stop_move(self):
        self.vel = 0
        self.rotation_vel = 0

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal
        self.angle -= self.rotation_vel

    # def collide(self, mask, x=0, y=0):
    #     car_mask = pygame.mask.from_surface(self.img)
    #     offset = (int(self.x - x), int(self.y - y))
    #     poi = mask.overlap(car_mask, offset)
    #     return poi
    
    def isCollide(self, win, obstacle):
        rect = self.img.get_rect()
        if rect.colliderect(obstacle):
            pygame.draw.rect(win, const.color["RED"], rect, 4)

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (180, 200)


def draw(win, images, player_car, game_info):
    for img in images:
        pygame.draw.rect(win, const.color["BLACK"], img)

    # level_text = MAIN_FONT.render(
    #     f"Level {game_info.level}", 1, const.color["BLACK"])
    # win.blit(level_text, (10, HEIGHT - level_text.get_height() - 70))

    time_text = MAIN_FONT.render(
        f"Time: {game_info.get_level_time()}s", 1, const.color["BLACK"])
    win.blit(time_text, (10, HEIGHT - time_text.get_height() - 40))

    vel_text = MAIN_FONT.render(
        f"Vel: {round(player_car.vel, 1)}px/s", 1, const.color["BLACK"])
    win.blit(vel_text, (10, HEIGHT - vel_text.get_height() - 10))

    player_car.draw(win)
    pygame.display.update()


def move_player(player_car):
    keys = pygame.key.get_pressed()

    if keys[pygame.K_a]:
        player_car.rotate(left=True)
    if keys[pygame.K_d]:
        player_car.rotate(right=True)
    if keys[pygame.K_w]:
        player_car.move_forward()
    if keys[pygame.K_s]:
        player_car.stop_move()    
        
    player_car.move()

# def handle_collision(player_car, computer_car, game_info):
#     if player_car.collide(TRACK_BORDER_MASK) != None:
#         player_car.bounce()

#     computer_finish_poi_collide = computer_car.collide(
#         FINISH_MASK, *FINISH_POSITION)
#     if computer_finish_poi_collide != None:
#         blit_text_center(WIN, MAIN_FONT, "You lost!")
#         pygame.display.update()
#         pygame.time.wait(5000)
#         game_info.reset()
#         player_car.reset()
#         computer_car.reset()

#     player_finish_poi_collide = player_car.collide(
#         FINISH_MASK, *FINISH_POSITION)
#     if player_finish_poi_collide != None:
#         if player_finish_poi_collide[1] == 0:
#             player_car.bounce()
#         else:
#             game_info.next_level()
#             player_car.reset()
#             computer_car.next_level(game_info.level)


run = True
clock = pygame.time.Clock()
images = [LEFT_LINE, RIGHT_LINE]
player_car = PlayerCar(4, 4)
game_info = GameInfo()

while run:
    clock.tick(FPS)
    WIN.fill(const.color["WHITE"])
    draw(WIN, images, player_car, game_info)

    while not game_info.started:
        blit_text_center(
            WIN, MAIN_FONT, f"Press any key to start level {game_info.level}!")
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break

            if event.type == pygame.KEYDOWN:
                game_info.start_level()
                
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    move_player(player_car)

    # handle_collision(player_car, computer_car, game_info)

    if game_info.game_finished():
        blit_text_center(WIN, MAIN_FONT, "You won the game!")
        pygame.time.wait(5000)
        game_info.reset()
        player_car.reset()


pygame.quit()