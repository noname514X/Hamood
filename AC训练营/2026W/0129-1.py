x = int(input())

left = 0
right = x
result = 0

while left <= right:
    mid = (left + right) // 2
    if mid * mid == x:
        result = mid
        break
    elif mid * mid < x:
        result = mid
        left = mid + 1
    else:
        right = mid - 1

print(result)
