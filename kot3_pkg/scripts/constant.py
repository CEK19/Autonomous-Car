#! /usr/bin/env python
import numpy as np

TOPIC_NAME_LANE_DETECTION = 'lane_detection_topic'
TOPIC_NAME_AVOIDANCE = 'avoidance_topic'
TOPIC_NAME_LIDAR = '/scan'
TOPIC_NAME_ACTION_DECISION = 'action_decision'


NODE_NAME_AVOIDANCE = 'avoidance_node_name'

class ASSETS:
    Q_TABLE_DATA = "./assets/QTable.txt"

class EXCEPTION:
    NO_Q_TABLE = "Dont have Q table file"

class MODULE_AVOIDANCE:
    LIST_LEVEL_ANGLES = np.array(range(10), dtype="str")