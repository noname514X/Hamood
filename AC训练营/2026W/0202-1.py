import ast

class Solution:
    def singleNumber(self, nums):
        res = 0
        for x in nums:
            res ^= x
        return res

line = input().strip()
if '=' in line:
    s = line.split('=', 1)[1].strip()
else:
    s = line
nums = ast.literal_eval(s)
print(Solution().singleNumber(nums))
