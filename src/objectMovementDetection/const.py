import math


class GameSettingParam:
    CAPTION = "Reinforcement Learning"
    WIDTH = 400
    # HEIGHT = 1000
    HEIGHT = 750
    FPS = 30


class PlayerParam:
    RADIUS_OBJECT = 10

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
    DO_NOTHING_VELO = "DO_NOTHING_VELO"

    INFINITY = 9999


class ObstacleParam:
    NUMBER_OF_OBSTACLES = 20
    OBSTACLE_ACCELERATION_FORWARD = 0 # 50
    OBSTACLE_ACCELERATION_ROTATE = 0 # 0.5
    MAX_VELOCITY = 70
    INITIAL_OBSTACLE_X = GameSettingParam.WIDTH//2
    INITIAL_OBSTACLE_Y = 0

    PROBABILITIES_ACTION = [0.08,
                            0.08,
                            0.08,
                            0.4,
                            0.28,
                            0.08]


class RLParam:

    MIN_EPSILON = 0
    MAX_EPSILON = 0.5

    MIN_ALPHA = 0.3
    MAX_ALPHA = 0.1

    GAMMA = 0.5

    AREA_RAY_CASTING_NUMBERS = 6

    N_EPISODES = 1000
    MAX_EPISODE_STEPS = 100000

    ACTIONS = [PlayerParam.INC_ROTATION_VELO,
               PlayerParam.DESC_ROTATION_VELO,
               PlayerParam.STOP,
               PlayerParam.INC_FORWARD_VELO,
               PlayerParam.DESC_FORWARD_VELO,
               PlayerParam.DO_NOTHING_VELO]

    DISTANCE_OF_RAY_CASTING = [
        # 0,
        int(PlayerParam.RADIUS_LIDAR*1/3),
        int(PlayerParam.RADIUS_LIDAR*2/3),
        PlayerParam.RADIUS_LIDAR,
        PlayerParam.INFINITY
    ]
    MAX_TIME_MS = 2*60

    class LEVEL_OF_RAY_CASTING:
        INFINITY = "3"  # NO TOUCH OBSTACLE
        # FAR_DISTANCE = "3"
        SAFETY_DISTANCE = "2"  # LIDAR TOUCH OBSTACLE, BUT SAFE
        DANGEROUS_DISTANCE = "1"  # LIDAR TOUCH OBSTACLE, BUT IN DANGEROUS MODE
        FAILED_DISTANCE = "0"  # LIDAR TOUCH OBSTACLE, AND OUCH

    class LEVEL_OF_LANE:
        # | x | 3x | 2x | 3x | x |
        # split the middle area into 2 parts, each part will be x    
        
        LEFT = "4"
        DISTANCE_LEFT = GameSettingParam.WIDTH * 4 / 10
        
        MOST_LEFT = "3"
        DISTANCE_MOST_LEFT =  GameSettingParam.WIDTH*1/10
        
        MIDDLE = "2"
        DISTANCE_MIDDLE = 0 # JUST LEAVE 0 FOR FUN
        
        
        RIGHT = "1"
        DISTANCE_RIGHT = GameSettingParam.WIDTH * 4 / 10
        
        MOST_RIGHT = "0"
        DISTANCE_MOST_RIGHT =  GameSettingParam.WIDTH*1/10

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
    
    class LEVEL_OF_ROTATION:
        MAX_LEFT = "0"
        MAX_LEFT_ANGLE = -20
        
        LEFT = "1"
        LEFT_ANGLE = -10
        
        CENTER = "2"
        CENTER_ANGLE = 0
        
        RIGHT = "3"
        RIGHT_ANGLE = 0
        
        MAX_RIGHT = "4"
        MAX_RIGHT_ANGLE = 10
        
        LIST_LEVEL_OF_ROTATION = [MAX_LEFT, LEFT, CENTER, RIGHT, MAX_RIGHT]
    
    class SCORE:
        # lidar detect obstacle
        OBSTACLE_TOUCH = -1_000_000_000
        FAILED_DISTANCE_TOUCH = -15000
        DANGEROUS_ZONE_TOUCH = -5000
        SAFETY_ZONE_TOUCH = -1000

        # stay in middle of lane
        STAY_AT_CENTER_OF_LANE = 1000
        STAY_AT_LEFT_OR_RIGHT_OF_LANE = -100
        STAY_AT_MOSTLEFT_OR_MOSTRIGHT_OF_LANE = -10000
        
        # angle of car
        STAY_IN_FRONT = 1000
        STAY_IN_NORMAL_ANGLE = -100
        
        # Actions
        STOPS_TO_ENJOY = -10000
        TURN_AROUND = -2000000
        INCREASE_Y = -300000
        INCREASE_SPEED_FORWARD = 10000
        
        FINISH_LINE = 10000000
        TOUCH_BOTTOM = -2_000_000_000


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
    I_AM_A_ROBOT = "YOU NOT SMART ENOUGHT TOBE A ROBOT OK ???"


class GUI:
    DISPLAY = "DISPLAY"
    HIDDEN = "HIDDEN"


class FILE:
    PROGRESS = "progress.txt"
    PROGRESS_BACKUP = "progress-backup.txt"
    MODEL_SAVE = "rl-learning.txt"
    MODEL_SAVE_BACKUP = "rl-learning-backup.txt"
