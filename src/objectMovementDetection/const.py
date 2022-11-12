import math


class GameSettingParam:
    CAPTION = "Reinforcement Learning"
    WIDTH = 400
    # HEIGHT = 1000
    HEIGHT = 750
    FPS = 30


class PlayerParam:
    RADIUS_OBJECT = 20

    ACCELERATION_FORWARD = 10
    ACCELERATION_ROTATE = 0.05

    WIDTH = 16
    HEIGHT = 30

    INITIAL_X = GameSettingParam.WIDTH//2
    INITIAL_Y = GameSettingParam.HEIGHT

    MAX_VELOCITY = 60
    MAX_ROTATION_VELOCITY = 20

    FOV = math.pi
    HALF_FOV = FOV/2
    CASTED_RAYS = 90
    STEP_ANGLE = FOV / CASTED_RAYS
    RADIUS_LIDAR = 140

    INC_ROTATION_VELO = "INC_ROTATION_VELO"
    DESC_ROTATION_VELO = "DESC_ROTATION_VELO"
    STOP = "STOP"
    INC_FORWARD_VELO = "INC_FORWARD_VELO"
    DESC_FORWARD_VELO = "DESC_FORWARD_VELO"

    INFINITY = 9999


class ObstacleParam:
    NUMBER_OF_OBSTACLES = 10
    OBSTACLE_ACCELERATION_FORWARD = 50
    OBSTACLE_ACCELERATION_ROTATE = 0.5
    MAX_VELOCITY = 70
    INITIAL_OBSTACLE_X = GameSettingParam.WIDTH//2
    INITIAL_OBSTACLE_Y = 0

    PROBABILITIES_ACTION = [0.1,
                            0.1,
                            0.1,
                            0.4,
                            0.3]


class RLParam:

    MIN_EPSILON = 0
    MAX_EPSILON = 0.7

    MIN_ALPHA = 1
    MAX_ALPHA = 0.1

    GAMMA = 0.5

    AREA_RAY_CASTING_NUMBERS = 6

    N_EPISODES = 1000
    MAX_EPISODE_STEPS = 100000

    ACTIONS = [PlayerParam.INC_ROTATION_VELO,
               PlayerParam.DESC_ROTATION_VELO,
               PlayerParam.STOP,
               PlayerParam.INC_FORWARD_VELO,
               PlayerParam.DESC_FORWARD_VELO]

    DISTANCE_OF_RAY_CASTING = [
        int(PlayerParam.RADIUS_LIDAR*1/3),
        int(PlayerParam.RADIUS_LIDAR*2/3),
        PlayerParam.RADIUS_LIDAR,
        PlayerParam.INFINITY
    ]
    MAX_TIME_MS = 2*60

    class LEVEL_OF_RAY_CASTING:
        INFINITY = "3"  # NO TOUCH OBSTACLE
        SAFETY_DISTANCE = "2"  # LIDAR TOUCH OBSTACLE, BUT SAFE
        DANGEROUS_DISTANCE = "1"  # LIDAR TOUCH OBSTACLE, BUT IN DANGEROUS MODE
        FAILED_DISTANCE = "0"  # LIDAR TOUCH OBSTACLE, AND OUCH

    # | x | 3x | 2x | 3x | x |
    # split the middle area into 2 parts, each part will be x
    DISTANCE_FROM_CENTER_OF_LANE = [
        # most left or most right (distance > 4x/10) (1/2 of middle + 3x area)
        GameSettingParam.WIDTH * 4 / 10,
        # left or right (distance > x/10) (1/2 of middle)
        GameSettingParam.WIDTH * 1 / 10,
        # center (distance > 0) (inside 1/2 of middle)
        0
    ]  # 0 is lucky number, no meaning

    class LEVEL_OF_LANE:
        LEFT = "4"
        MOST_LEFT = "3"
        MIDDLE = "2"
        RIGHT = "1"
        MOST_RIGHT = "0"

        LIST_LEVEL_OF_LANE = [LEFT, MOST_LEFT, MIDDLE, RIGHT, MOST_RIGHT]

    class LEVEL_OF_ANGLE:
        FRONT = "0"
        FRONT_ANGLE = math.pi
        
        
        NORMAL_LEFT = "1"
        NORMAL_LEFT_ANGLE = math.pi - math.pi/4
         
        NORMAL_RIGHT = "2"
        NORMAL_RIGHT_ANGLE = math.pi + math.pi/4
        
        OVER_ROTATION_LEFT = "3"
        OVER_ROTATION_LEFT_ANGLE = math.pi - math.pi/2
        
        OVER_ROTATION_RIGHT = "4"
        OVER_ROTATION_RIGHT_ANGLE = math.pi + math.pi/2
        
        LIST_LEVEL_ANGLES = [FRONT, NORMAL_LEFT, NORMAL_RIGHT, OVER_ROTATION_LEFT, OVER_ROTATION_RIGHT]
    class SCORE:
        # lidar detect obstacle
        OBSTACLE_TOUCH = -10000000
        DANGEROUS_ZONE_TOUCH = -10000000

        # stay in middle of lane
        STAY_AT_CENTER_OF_LANE = 1000
        STAY_AT_LEFT_OR_RIGHT_OF_LANE = -500
        STAY_AT_MOSTLEFT_OR_MOSTRIGHT_OF_LANE = -10000

        # Actions
        STOPS_TO_ENJOY = -10000
        TURN_AROUND = -10000000
        INCREASE_Y = 10000
        INCREASE_SPEED_FORWARD = 10000


class CustomColor:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    CYAN = (0, 255, 255)
    PINK = (255, 0, 255)


class MODE_PLAY:
    MANUAL = "MANUAL"
    RL_TRAIN = "RL_TRAIN"
    RL_DEPLOY = "RL_DEPLOY"


class GUI:
    DISPLAY = "DISPLAY"
    HIDDEN = "HIDDEN"


class FILE:
    PROGRESS = "progress.txt"
    PROGRESS_BACKUP = "progress-backup.txt"
    MODEL_SAVE = "rl-learning.txt"
    MODEL_SAVE_BACKUP = "rl-learning-backup.txt"
