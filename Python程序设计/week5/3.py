from random import randint
x = [randint(1,100) for i in range(20)]
print(x)
sorted_x = sorted(x, key = lambda item:item%2==0)
print(sorted_x)