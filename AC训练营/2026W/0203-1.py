import sys
import ast

data=sys.stdin.read().strip()
if '=' in data:
    data=data.split('=',1)[1]
nums=ast.literal_eval(data)

l=0
r=len(nums)-1
while l<r:
    m=(l+r)//2
    if nums[m]>nums[r]:
        l=m+1
    else:
        r=m
print(nums[l])
