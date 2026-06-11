import sys
import re
import ast

def search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return m
        if nums[l] <= nums[m]:
            if nums[l] <= target < nums[m]:
                r = m - 1
            else:
                l = m + 1
        else:
            if nums[m] < target <= nums[r]:
                l = m + 1
            else:
                r = m - 1
    return -1


def main():
    data = sys.stdin.read()
    m_list = re.search(r"\[[\s\S]*?\]", data)
    nums = []
    if m_list:
        try:
            nums = ast.literal_eval(m_list.group(0))
        except Exception:
            nums = []
    m_target = re.search(r"target\s*=\s*(-?\d+)", data)
    if m_target:
        target = int(m_target.group(1))
    else:
        rest = data
        if m_list:
            rest = data.replace(m_list.group(0), " ")
        m_num = re.search(r"-?\d+", rest)
        if m_num:
            target = int(m_num.group(0))
        else:
            target = 0
    print(search(nums, target))


if __name__ == '__main__':
    main()
