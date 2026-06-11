import sys
import re
import ast

def calculate(s: str) -> int:
    stack = []
    res = 0
    sign = 1
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isdigit():
            num = 0
            while i < n and s[i].isdigit():
                num = num * 10 + int(s[i])
                i += 1
            res += sign * num
            continue
        elif c == '+':
            sign = 1
        elif c == '-':
            sign = -1
        elif c == '(':
            stack.append((res, sign))
            res = 0
            sign = 1
        elif c == ')':
            prev_res, prev_sign = stack.pop()
            res = prev_res + prev_sign * res
        i += 1
    return res


def main():
    data = sys.stdin.read()
    m = re.search(r"([\"\'])(.*?)\1", data, re.S)
    if m:
        s = m.group(2)
    else:
        s = data.strip()
    print(calculate(s))


if __name__ == '__main__':
    main()
