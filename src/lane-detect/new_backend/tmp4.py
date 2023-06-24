import socket
import time
import cv2

tmp = {"tl": [-451, 893], "bl": [539, -843], "tr": [-222, -976], "br": [327, 946]}

print(cv2.clipLine((0,0,128,128),tmp["tl"],tmp["bl"]))