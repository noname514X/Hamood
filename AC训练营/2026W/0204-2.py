n = int(input().strip())
res = 0
p = 5
while n // p:
    res += n // p
    p *= 5
print(res)
