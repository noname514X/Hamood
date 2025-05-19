import collections
x = list(range(20))


def f(x):
    x = collections.deque(x)
    return x.rotate(-3)

d = f(x)
d.rotate(-3)
print(d)