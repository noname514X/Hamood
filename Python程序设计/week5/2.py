import random

data1 = [random.randint(1,10)] * 5
data2 = random.choices(range(10), k=5)
data3 = random.sample(range(10), k=5)

data0 = [1,1,4,5,1,4]
data00 = set(data0)
print(data0)
print(data00)


for data in (data1, data2, data3):
    print('='*20)
    print(data)
    k1 = len(set(data))
    k2 = len(data)
    if k1 == k2:
        print('无重复')
    elif k1 == 1:
        print('完全重复')
    else:
        print('部分重复')