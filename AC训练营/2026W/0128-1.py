import sys
import re
import ast

def removeDuplicates(nums):
    if not nums:
        return 0
    i = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    return i + 1

def main():
    data = sys.stdin.read()
    m = re.search(r"\[[\s\S]*?\]", data)
    nums = []
    if m:
        try:
            nums = ast.literal_eval(m.group(0))
        except Exception:
            nums = []
    k = removeDuplicates(nums)
    print(k)
    print(nums[:k])

if __name__ == '__main__':
    main()
