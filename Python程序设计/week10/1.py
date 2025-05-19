from math import pi as PI
import time
def circle_area(r):
    return PI * r ** 2
print(circle_area(114514)) 

start = time.time()
def demo(t):
    print([1])
    print([1,1])
    line = [1,1]
    for i in range(2,t):
        r = []
        for j in range(0, len(line) - 1):
            r.append(line[j]+line[j+1])
        line = [1] + r + [1]
        print(line)
demo(114514)
print(time.time() - start,"秒")