class Mode:
    ROS = 0
    PICS_IN_FOLDER = 1
    PIC = 2
    CAMERA = 3
    VIDEO = 4
    STORE_DATA = 5

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
    MODE = Mode.VIDEO
    ENABLE_WRITE_FILE = False
    MODEL_NAME = "model-110.h5"
    MODEL_PATH = "./models"
    VIDEO_PATH = "/Users/lap15864-local/Desktop/demo/orgVid.mp4"