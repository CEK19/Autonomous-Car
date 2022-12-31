import numpy as np

class Mode:
	FILE = 'file'
	PIC = 'pic'
	CAMERA = 'camera'

class TRAFFIC_LIGHT:
	red = "red"
	yellow = "yellow"
	green = "green"


class COLOR_THRESHOLD:
	pic = {
		"yellow": {
			"sensitivity": np.array([]),
			"lower": np.array([10, 135, 155]),	# H,S,V
			"upper": np.array([25, 255, 255])
		},
		"red": {
			"sensitivity": np.array([170, 0, 0]),
			"lower": np.array([0, 113, 150]),
			"upper": np.array([10, 255, 255])
		},
		"green": {
			"sensitivity": np.array([]),
			"lower": np.array([80, 50, 115]),
			"upper": np.array([90, 255, 255])
		}
	},
	camera = {
		"yellow": {
			"sensitivity": np.array([]),
			"lower": np.array([10, 135, 155]),	# H,S,V
			"upper": np.array([25, 255, 255])
		},
		"red": {
			"sensitivity": np.array([170, 0, 0]),
			"lower": np.array([0, 113, 150]),
			"upper": np.array([10, 255, 255])
		},
		"green": {
			"sensitivity": np.array([]),
			"lower": np.array([80, 50, 115]),
			"upper": np.array([90, 255, 255])
		}
	}


class STANDARD_PROPERTY:
	class pic:
		minArea = 20
		maxArea = 1000
		widthHeightRatio = 1.15
	class camera:
		minArea = 100
		maxArea = 1000
		widthHeightRatio = 1.15
	
# BGR
class COLOR:
	green = (0, 255, 0)
	red = (0, 0, 255)
	yellow = (0, 255, 255)
	
	
class Setting:
	PICTURE_PATH = "./assets/yellow2.jpeg"
	MODE = Mode.CAMERA
	COLOR_THRESHOLD = COLOR_THRESHOLD.camera
	STANDARD_PROPERTY = STANDARD_PROPERTY.camera