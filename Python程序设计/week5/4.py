import heapq
import random
data = list(range(10))
print("data:",data)
print("random delected data:", random.choice(data))
random.shuffle(data)
print("data:",data)

heap = []
for n in data:
    heapq.heappush(heap,n)
print("heap:",heap)

heapq.heappush(heap,0.5)
print(heap)
heapq.heappop(heap)
print(heap)