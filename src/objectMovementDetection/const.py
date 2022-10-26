import imp


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
    
    OBSTACLE_ACCELERATION_FORWARD = 10
    OBSTACLE_ACCELERATION_ROTATE = 0.05
    
    WIDTH = 16
    HEIGHT = 30
    
    INITIAL_X = 200
    INITIAL_Y = 200
    
    INITIAL_OBSTACLE_X = GameSettingParam.WIDTH//2
    INITIAL_OBSTACLE_Y = GameSettingParam.HEIGHT//2  
    
    MAX_VELOCITY = 200
    MAX_ROTATION_VELOCITY = 20
    
    FOV = math.pi/2
    HALF_FOV = FOV/2
    CASTED_RAYS = 90
    STEP_ANGLE = FOV / CASTED_RAYS
    RADIUS_LIDAR = 100
    
    INC_ROTATION_VELO = "INC_ROTATION_VELO"
    DESC_ROTATION_VELO="DESC_ROTATION_VELO"
    STOP="STOP"
    INC_FORWARD_VELO="INC_FORWARD_VELO"
    DESC_FORWARD_VELO="DESC_FORWARD_VELO"
                    

class CustomColor:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
