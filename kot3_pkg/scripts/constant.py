#! /usr/bin/env python
import numpy as np

TOPIC_NAME_LANE_DETECTION = 'lane_detection_topic'
TOPIC_NAME_AVOIDANCE = 'avoidance_topic'
TOPIC_NAME_CAMERA = '/camera/rgb/image_raw'
TOPIC_NAME_LIDAR = '/scan' # Check it
TOPIC_NAME_VELOCITY = '/cmd' # Check it
TOPIC_NAME_ACTION_DECISION = 'action_decision'
TOPIC_NAME_TRAFFIC_LIGHTS ='traffic_lights_topic'
TOPIC_NAME_TRAFFIC_SIGNS = 'traffic_signs_topic'


NODE_NAME_AVOIDANCE = 'avoidance_node_name'
NODE_NAME_TRAFFIC_SIGNS = 'traffic_signs_node_name'
NODE_NAME_TRAFFIC_LIGHTS = 'traffic_lights_node_name'
NODE_NAME_DECISION_MAKING = 'decision_making_node_name'

class ASSETS:
    Q_TABLE_DATA = "./assets/QTable.txt"

class EXCEPTION:
    NO_Q_TABLE = "Dont have Q table file"

class MODULE_AVOIDANCE:
    LIST_LEVEL_ANGLES = np.array(range(10), dtype="str")

    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    GO_AHEAD = "GO_AHEAD"
    GO_BACKWARD = "GO_BACKWARD"

class MODULE_TRAFFIC_LIGTHS:
    RED = "RED"
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    
class MODULE_TRAFFIC_SIGNS:
    AHEAD = "AHEAD"
    FORBID = "FORBID"
    STOP = "STOP"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    NONE = "NONE"
    LABEL_TO_TEXT = [AHEAD, FORBID, STOP, LEFT, RIGHT, NONE]



#--------------#
#      Tú      #
#--------------#
class Sign:
    EXTRA_SAFETY = 20
    MIN_AREA = 100
    MAX_AREA = 50000
    MIN_WIDTH_HEIGHT = 30
    MIN_ACCURACY = 0.2
    

# class Text:
#     FONT_SIZE = 1.5
#     FIRST_LINE_ORG = (10,40)
#     SECOND_LINE_ORG = (10,90)
#     THICKNESS = 3

## for window
class Text:
    FONT_SIZE = 0.5
    FIRST_LINE_ORG = (10,25)
    SECOND_LINE_ORG = (10,50)
    THICKNESS = 2


###########################

class Setting:
    MODEL_NAME = "model-110.h5"
    MODEL_PATH = "./models"