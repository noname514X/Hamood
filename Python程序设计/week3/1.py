#zip
def zipf():
    a = [1,2,3,4]
    b = ['a','b','c','d']
    c = zip(a,b)

    for (a,b) in zip(a,b):
        print(a,b)


    a = list(range(10))
    print(a[1:4])
#切片操作p31
def p31():
    aList = [1,2,3,4,5,6,7,8,9,10]
    print(aList[::])
    print(aList[::-1])
    print(aList[::2])
    print(aList[1::2])
    print(aList[3::])
    print(aList[3::6])
    print(aList[0:100:1])
    print(aList[100::])
#切片操作-原地修改 p32
def p32():
    aList = [1,2,3,4,5,6,7,8,9,10]
    aList[len(aList):] = [11]
    print(aList)
    aList[0:3] = [-1,-2,-3]
    print(aList)
    aList[0:3] = []
    print(aList)
    aList[::2] = [0]*4
    print(aList)
    aList[1::2] = [6]*4
    print(aList)
#列表排序
def aaa():
    import random
    def s_len(x):
        return len(str(x))
    a = list(range(20))
    random.shuffle(a)
    print(a)
    a.sort(key = lambda x:len(str(x)))
    print(a)

    print(sorted(a))
    print(sorted(a,reverse=True))

def bbb():
    def s_len(x):
        return len(str(x))

    bList = ['hamood','ccb','a','chushen']
    print(bList.sort(key = len))
    for i in bList():
        print(i)

import os
filea = [filename for filename in os.listdir('.') if filename.endswith('.py')]
print(filea)

