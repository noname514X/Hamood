import sys
import re
import ast


def containsNearbyDuplicate(nums: list, k: int) -> bool:
    seen = {}
    for i, v in enumerate(nums):
        if v in seen and i - seen[v] <= k:
            return True
        seen[v] = i
    return False


def main():
    data = sys.stdin.read()
    m_list = re.search(r"\[[\s\S]*?\]", data)
    nums = []
    if m_list:
        try:
            nums = ast.literal_eval(m_list.group(0))
        except Exception:
            nums = []
    m_k = re.search(r"k\s*=\s*(-?\d+)", data)
    if m_k:
        k = int(m_k.group(1))
    else:
        rest = data
        if m_list:
            rest = data.replace(m_list.group(0), " ")
        m_num = re.search(r"-?\d+", rest)
        if m_num:
            k = int(m_num.group(0))
        else:
            k = 0
    print("true" if containsNearbyDuplicate(nums, k) else "false")


if __name__ == '__main__':
    main()
