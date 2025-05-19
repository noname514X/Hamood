import time
start = time.time()

n = int(input())
for num in range(10**(n-1),10**n):
    if sum(map(lambda i: int(i)**n, str(num))) == num:
        print(num)

print(time.time()-start)