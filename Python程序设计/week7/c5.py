from itertools import count
for num in count(16,9):
    if num%5 == 2:
        break
for result in count(num,45):
    if result%4 == 1:
        break
print(result)


