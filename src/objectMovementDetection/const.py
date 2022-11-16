import math


class GameSettingParam:
    CAPTION = "Reinforcement Learning"
    WIDTH = 400
    # HEIGHT = 1000
    HEIGHT = 750
    FPS = 30
    DRAW = False
    
    class EndGameReason:
        NOT_END_GAME = ""
        WIN = "win"
        TOUCH_OBSTACLE = "touch_obstacle"
        TOUCH_SIDE = "touch_side"
        TOUCH_BOTTOM = "touch_bottom"
        OVER_ROTAION = "over_rotation"


class PlayerParam:
    RADIUS_OBJECT = 7

    ACCELERATION_FORWARD = 10
    ACCELERATION_ROTATE = 0.05

    WIDTH = 16
    HEIGHT = 30

    INITIAL_X = GameSettingParam.WIDTH//2
    INITIAL_Y = GameSettingParam.HEIGHT - 20

    MAX_VELOCITY = 80
    MAX_ROTATION_VELOCITY = 5

    FOV = math.pi/3
    HALF_FOV = FOV/2
    CASTED_RAYS = 60
    STEP_ANGLE = FOV / CASTED_RAYS
    RADIUS_LIDAR = 160

    INC_ROTATION_VELO = "INC_ROTATION_VELO"
    DESC_ROTATION_VELO = "DESC_ROTATION_VELO"
    DO_NOTHING = "DO_NOTHING"
    INC_FORWARD_VELO = "INC_FORWARD_VELO"
    DESC_FORWARD_VELO = "DESC_FORWARD_VELO"

    INFINITY = 9999


class ObstacleParam:
    NUMBER_OF_OBSTACLES = 8
    OBSTACLE_ACCELERATION_FORWARD = 50
    OBSTACLE_ACCELERATION_ROTATE = 0.5
    MAX_VELOCITY = 0
    INITIAL_OBSTACLE_X = GameSettingParam.WIDTH//2
    INITIAL_OBSTACLE_Y = 0

    PROBABILITIES_ACTION = [0.1,
                            0.1,
                            0.1,
                            0.4,
                            0.3]


class RLParam:

    MIN_EPSILON = 0
    MAX_EPSILON = 0.5

    MIN_ALPHA = 0.5
    MAX_ALPHA = 0.1

    GAMMA = 0.75

    AREA_RAY_CASTING_NUMBERS = 4

    N_EPISODES = 3000
    N_EPISODES_PER_SAVE_MODEL = 100
    MAX_EPISODE_STEPS = 100000

    ACTIONS = [PlayerParam.INC_ROTATION_VELO,
               PlayerParam.DESC_ROTATION_VELO,
               PlayerParam.DO_NOTHING,
               PlayerParam.INC_FORWARD_VELO,
               PlayerParam.DESC_FORWARD_VELO]

    # DISTANCE_OF_RAY_CASTING = [
    #     int(PlayerParam.RADIUS_LIDAR*1/3),
    #     int(PlayerParam.RADIUS_LIDAR*2/3),
    #     PlayerParam.RADIUS_LIDAR,
    #     PlayerParam.INFINITY
    # ]
    
    DISTANCE_OF_RAY_CASTING = [
        PlayerParam.RADIUS_OBJECT + 10,
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
        
    class SCORE:
        # lidar detect obstacle
        # OBSTACLE_TOUCH = -1000000
        
        # FAILED_DISTANCE_TOUCH = -1000000 # In use
        # DANGEROUS_ZONE_TOUCH = 0.1 # In use
        # SAFETY_ZONE_TOUCH = 0.01 # In use
        RAY_CAST_COST = -5
        
        TOUCH_BOTTOM = -1000000 # In use

        # stay in middle of lane
        STAY_AT_CENTER_OF_LANE = 0.01
        STAY_AT_LEFT_OR_RIGHT_OF_LANE = -2
        STAY_AT_MOSTLEFT_OR_MOSTRIGHT_OF_LANE = -100
        
        # angle of car
        STAY_IN_FRONT = 1000
        STAY_IN_NORMAL_ANGLE = -100        
                        
        FINISH_LINE = 10000
        
        # rotation
        OVER_ROTATION = -1000000


class CustomColor:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    CRYAN = (0, 255, 255)
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

class Equation:
    NO_SOLUTION = 0
    ONE_SOLUTION = 1
    TWO_SOLUTION = 2