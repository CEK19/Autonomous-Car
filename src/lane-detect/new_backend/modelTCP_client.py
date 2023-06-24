import socket
import cv2
import numpy as np
import time

HOST = '192.168.88.149'  
PORT = 8000        

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
print('connecting to %s port ' + str(server_address))
s.connect(server_address)
cam = cv2.VideoCapture(0)

try:
    while True:
        _,frame = cam.read()
        frame = cv2.resize(frame,(128,128))
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

        start = time.time()
        msg = frame.tobytes()
        s.sendall(msg)
        print("send done")
        data = s.recv(1024)
        data = float(data.decode("utf8"))
        print("total time:",time.time()-start,"transfer time:",time.time()-start-data)
        cv2.waitKey(1000)
finally:
    s.close()