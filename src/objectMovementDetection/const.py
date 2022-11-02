import math


class GameSettingParam:
    CAPTION = "Reinforcement Learning"
    WIDTH = 400
    HEIGHT = 1000
    FPS = 60


class PlayerParam:
    RADIUS_OBJECT = 20

    ACCELERATION_FORWARD = 10
    ACCELERATION_ROTATE = 0.05

    WIDTH = 16
    HEIGHT = 30

    INITIAL_X = GameSettingParam.WIDTH//2
    INITIAL_Y = GameSettingParam.HEIGHT

    MAX_VELOCITY = 200
    MAX_ROTATION_VELOCITY = 20

    FOV = math.pi/2
    HALF_FOV = FOV/2
    CASTED_RAYS = 90
    STEP_ANGLE = FOV / CASTED_RAYS
    RADIUS_LIDAR = 100

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
    INITIAL_OBSTACLE_X = GameSettingParam.WIDTH//2
    INITIAL_OBSTACLE_Y = 0
    
    PROBABILITIES_ACTION = [0.1,
                            0.1,
                            0.1,
                            0.4,
                            0.3]

class RLParam:
    EPSILON = 0.2

    MAX_ALPHA = 0.1
    MIN_ALPHA = 1

    GAMMA = 0.6

    AREA_RAY_CASTING_NUMBERS = 10

    N_EPISODES = 20
    MAX_EPISODE_STEPS = 100

    ACTIONS = [PlayerParam.INC_ROTATION_VELO,
               PlayerParam.DESC_ROTATION_VELO,
               PlayerParam.STOP,
               PlayerParam.INC_FORWARD_VELO,
               PlayerParam.DESC_FORWARD_VELO]

    DISTANCE_OF_RAY_CASTING = [10, 20, 30, 9999999] # 9999999 is infinity
    
    class LEVEL_OF_RAY_CASTING:
        INFINITY = "3" # NO TOUCH OBSTACLE 
        SAFETY_DISTANCE = "2" # LIDAR TOUCH OBSTACLE, BUT SAFE
        DANGEROUS_DISTANCE = "1" # LIDAR TOUCH OBSTACLE, BUT IN DANGEROUS MODE
        FAILED_DISTANCE = "0" # LIDAR TOUCH OBSTACLE, AND OUCH
        
    DISTANCE_FROM_CENTER_OF_LANE = [20, 10, -9999] # -9999 is infinity
    
    class LEVEL_OF_LANE:
        LEFT = "4"
        MOST_LEFT = "3"
        MIDDLE =  "2"
        RIGHT = "1"
        MOST_RIGHT = "0"


class CustomColor:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)


class MODE_PLAY:
    MANUAL = "MANUAL"
    RL_TRAIN = "RL_TRAIN"


class GUI:
    DISPLAY = True
    HIDDEN = False
