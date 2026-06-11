import sys
import re
import ast


def min_subarray_len(target: int, nums: list) -> int:
    n = len(nums)
    left = 0
    cur = 0
    ans = n + 1
    for right in range(n):
        cur += nums[right]
        while cur >= target:
            ans = min(ans, right - left + 1)
            cur -= nums[left]
            left += 1
    return ans if ans <= n else 0


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
    print(min_subarray_len(target, nums))


if __name__ == '__main__':
    main()
