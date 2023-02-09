import numpy as np

class Mode:
	FILE = 'file'
	PIC = 'pic'
	CAMERA = 'camera'
	VIDEO = 'video'

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
	}
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
	demo = {
		"yellow": {	# done
			"sensitivity": np.array([]),
			"lower": np.array([15, 20, 230]),	# H,S,V
			"upper": np.array([30, 255, 255])
		},
		"red": {	# demo
			"sensitivity": np.array([170, 0, 0]),		# [170, 0, 0]
			"lower": np.array([0, 180, 110]),	# [0, 210, 110]	[0, 228, 124]
			"upper": np.array([7, 255, 255])
		},
		"green": {	# done
			"sensitivity": np.array([]),
			"lower": np.array([48, 119, 125]),
			"upper": np.array([93, 255, 215])
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

	class demo:
		minArea = 100
		maxArea = 10000
		widthHeightRatio = 140
	
# BGR
class COLOR:
	green = (0, 255, 0)
	red = (0, 0, 255)
	yellow = (0, 255, 255)
	white = (255, 255, 255)
	
	
class Setting:
	PICTURE_PATH = "./assets/demo/red2.jpg"	# "C:\\Users\\Admin\\Documents\\coding\\Autonomous-Car\\src\\trafficLightDetection\\assets\\demo\\red2.jpg"
	VIDEO_PATH = "./assets/ignore/yellowVid.mp4"	# PATH = "./assets/ignore/org.mp4"
	MODE = Mode.VIDEO
	COLOR_THRESHOLD = COLOR_THRESHOLD.demo
	STANDARD_PROPERTY = STANDARD_PROPERTY.demo

# "./assets/ignore/redVid.mp4"
# "./assets/ignore/yellowVid.mp4"
# "./assets/ignore/greenVidFinal.mov"