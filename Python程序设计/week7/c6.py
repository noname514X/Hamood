from itertools import combinations_with_replacement

primes = (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97)
for item in combinations_with_replacement(primes,3):
    if sum(map(lambda x:x**2,item)) == 2019:
        print(item)