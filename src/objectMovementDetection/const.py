import math


class GAME_SETTING:
    SCREEN_WIDTH = 200
    SCREEN_HEIGHT = 400
    FPS = 10
    
    MODE_MANUALLY = "MODE_MANUALLY"
    MODE_AUTO = "MODE_AUTO"


class PLAYER_SETTING:
    RADIUS_OBJECT = 10
    RADIUS_LIDAR = 342  # From the border of the circle

    INITIAL_X = GAME_SETTING.SCREEN_WIDTH//2
    INITIAL_Y = GAME_SETTING.SCREEN_HEIGHT - 20

    MAX_FORWARD_VELO = 60
    MAX_ROTATION_VELO = 10
    MIN_ROTATION_VELO = -MAX_ROTATION_VELO

    ACCELERATION_FORWARD = 5
    ACCELERATION_ROTATE = 0.05

    CASTED_RAYS = 90
    FOV = math.pi
    HALF_FOV = FOV/2
    STEP_ANGLE = FOV / CASTED_RAYS

    Y_GOAL_POSITION = 10


class LANE_SETTING:
    WIDTH_OF_LANE_BORDER = 3

    LEFT_PADDING = 0
    RIGHT_PADDING = LEFT_PADDING
    TOP_PADDING = 0    


class OBSTACLE_SETTING:
    MAX_INSTANCES = 0
    RADIUS_OBJECT = 30
    PROBABILITIES_ACTION = [0.1,
                            0.1,
                            0.1,
                            0.4,
                            0.2,
                            0.1]


class COLOR:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    CYAN = (0, 255, 255)
    PINK = (255, 0, 255)


# RELATED TO REINFORCEMENT LEARNING
class ACTIONS:
    TURN_RIGHT_ACCELERATION = 0
    TURN_LEFT_ACCELERATION = 1
    STOP = 2
    FORWARD_ACCELERATION = 3
    BACKWARD_ACCELERATION = 4
    DO_NOTHING = 5  # DIFFERENT WITH STOP


class EQUATION:
    NO_SOLUTION = 0
    ONE_SOLUTION = 1
    TWO_SOLUTION = 2


ACTION_SPACE = 6
ACTIONS_LIST = [
    ACTIONS.TURN_RIGHT_ACCELERATION,
    ACTIONS.TURN_LEFT_ACCELERATION,
    ACTIONS.STOP,
    ACTIONS.FORWARD_ACCELERATION,
    ACTIONS.BACKWARD_ACCELERATION,
    ACTIONS.DO_NOTHING,
]
MAX_EPISODE = 100000
INT_INFINITY = 99999


class D_STAR:
    class ENV:
        START_POINT = (3, 5)
        GOAL_POINT = (10, 2)
        EPSILON = 2.5
        EPS_MIN = 1.0
        EPS_MINUS_PER_RUN = 0.5
        EPS_PLUS_PER_ENV_CHANGE = 0.5
        IS_PAUSE = False
        IS_PRINT = False
        IS_PLOTTING = False
        HAS_OBS = 1
        NO_OBS = 0

    MY_MAP = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    
    # MY_MAP = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    # ]    

    NEW_MAP = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    
    NEW_MAP_1 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    
    NEW_BLANK_MAP = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    NEW_BLOCK_MAP = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    
    MOVING_MAP = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 2, 2, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]