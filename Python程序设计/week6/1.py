import time
n = int(input())
multiples = set()
candidates = set(range(1, n))
start = time.time()


for i in range(14, n, 14):  
    multiples.add(i)

print(multiples)
print(time.time()-start)