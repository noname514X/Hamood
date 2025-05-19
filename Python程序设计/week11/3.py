def func(factors, x):
    result = factors[0]
    for factor in factors[1:]:
        result = result * x + factor
    return result

from functools import reduce

def func2(factors, x):
    result = reduce(lambda a, b: a*x+b, factors)
    return result

factors = (3,8,5,9,7,1)
print(func(factors, 1))
print(func2(factors, 2))
factors = (5,0,0,0,0,1)
print(func(factors, 2))
print(func2(factors, 3))    
