import numpy as np


class TRAFFIC_LIGHT:
    red = "red"
    yellow = "yellow"
    green = "green"
    
    
COLOR_THRESHOLD = {
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
    minArea = 20
    maxArea = 1000
    
# BGR
class COLOR:
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)