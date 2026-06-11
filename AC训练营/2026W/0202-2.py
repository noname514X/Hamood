import ast

class Solution:
    def singleNumber(self, nums):
        res = 0
        for i in range(32):
            s = 0
            for x in nums:
                s += (x >> i) & 1
            if s % 3:
                res |= (1 << i)
        if res >= 2 ** 31:
            res -= 2 ** 32
        return res

line = input().strip()
if '=' in line:
    s = line.split('=', 1)[1].strip()
else:
    s = line
nums = ast.literal_eval(s)
print(Solution().singleNumber(nums))
