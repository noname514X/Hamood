from random import sample
data = [sample(range(1,100), 10) for i in range(5)]
for row in data:
    print(row)
print()
for row in sorted(data):
    print(row)