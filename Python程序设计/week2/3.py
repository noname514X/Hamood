print(3,5,7, sep='#')
print(3,5,7, sep='\n')
print(3,5,7, sep=',\n')

import math
print(math.sin(114514))
print(math.cos(1919810))
print(math.tan(810893))

x = input("请输入一个三位数：")
a, b, c = map(int,x)
print(a,b,c)

x = input("请输入一个n位数：")
print(*map(int,x))