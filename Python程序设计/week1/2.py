import random
data = random.choices(range(10), k=10)

print(data)

data.sort()
print(data)

print(data[3])

print(data[1:5])

data.remove(8)

print(data)

data = ['red', 'Green', 'blue']
data.sort(key=str.lower)
print(data)