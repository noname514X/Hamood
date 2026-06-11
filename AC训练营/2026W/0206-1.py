import sys
import re
import ast

def is_happy(n: int) -> bool:
    def nxt(x: int) -> int:
        s = 0
        while x:
            d = x % 10
            s += d * d
            x //= 10
        return s
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = nxt(n)
    return n == 1


def main():
    data = sys.stdin.read()
    m = re.search(r"n\s*=\s*(-?\d+)", data)
    if m:
        n = int(m.group(1))
    else:
        try:
            val = ast.literal_eval(data.strip())
            if isinstance(val, dict) and 'n' in val:
                n = int(val['n'])
            elif isinstance(val, (list, tuple)):
                n = int(val[0])
            else:
                nums = re.findall(r"-?\d+", data)
                n = int(nums[0]) if nums else 0
        except Exception:
            nums = re.findall(r"-?\d+", data)
            n = int(nums[0]) if nums else 0
    print("true" if is_happy(n) else "false")


if __name__ == '__main__':
    main()
