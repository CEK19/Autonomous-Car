import json

test1 = {"abbc":  1, "xyz": "10"}

k = json.dumps(test1)
print(k)

t = json.loads(k)
print(t['abbc'])

from module import *
test()