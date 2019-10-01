import os
import time
import random

log = 'test'

print(os.path.exists(log))

dir = os.walk(log)

for d in dir:
    print(d)

print(str(int(time.time())))

a = list(range(10))
random.shuffle(a)
print(a)





