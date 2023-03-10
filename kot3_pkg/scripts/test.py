import json
import numpy as np

test1 = {"abbc":  1, "xyz": "10"}

k = json.dumps(test1)
print(k)

t = json.loads(k)
print(t['abbc'])
x = np.zeros(shape=(180, ))
print(x.shape)
ranges = [1, 2 ,3]
print(ranges[0:2])
convertedLidarSignalBaseAngle = np.zeros(shape=((180, )))
convertedLidarSignalBaseAngle[0:2] = ranges[0:2]
print(type(convertedLidarSignalBaseAngle))

x = np.arange(0, 10)
y = np.arange(0, 10)
print(y*np.cos(x))

t = [1, 2, 3]
print("aaaa", t[::-1][::-1])