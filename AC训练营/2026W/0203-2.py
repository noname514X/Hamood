import sys
import ast

class Solution:
    def findMin(self, nums):
        l=0
        r=len(nums)-1
        while l<r:
            m=(l+r)//2
            if nums[m]>nums[r]:
                l=m+1
            elif nums[m]<nums[r]:
                r=m
            else:
                r-=1
        return nums[l]

data=sys.stdin.read().strip()
if '=' in data:
    data=data.split('=',1)[1]
nums=ast.literal_eval(data)
print(Solution().findMin(nums))
