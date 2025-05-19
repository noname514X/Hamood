keys = ['a', 'b', 'c', 'd']
values = [1, 2, 3, 4]
for k,v in zip(keys, values):
    print((k,v), end=' ')
print()

x = ['a', 'b', 'c']
for i,v in enumerate(x):
    print('The value on position {0} is {1}' .format(i,v))