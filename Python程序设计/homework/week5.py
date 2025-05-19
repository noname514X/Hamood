#使用集合实现筛选法求素数。输入一个大于2的自然数，输出小于该数字的所有素数组成的集合。
#提示：请自学并使用集合的discard方法。
import time
n = int(input())
primes = set()
candidates = set(range(2, n))
start = time.time()

while candidates:
    p = min(candidates)
    primes.add(p)
    i = p
    while i < n:
        candidates.discard(i)
        i += p
    if p * p > n:
        primes.update(candidates)
        break

print(primes)
print(time.time()-start)