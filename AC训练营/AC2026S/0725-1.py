def myAtoi(s):
    i = 0
    n = len(s)
    while i < n and s[i] == ' ':
        i += 1
    if i == n:
        return 0
    sign = 1
    if s[i] == '-':
        sign = -1
        i += 1
    elif s[i] == '+':
        i += 1
    result = 0
    while i < n and s[i].isdigit():
        digit = ord(s[i]) - ord('0')
        if result > (2**31 - 1 - digit) // 10:
            return 2**31 - 1 if sign == 1 else -2**31
        result = result * 10 + digit
        i += 1
    return sign * result

if __name__ == "__main__":
    print(myAtoi("42"))
    print(myAtoi("   -42"))
    print(myAtoi("1337c0d3"))
    print(myAtoi("0-1"))
    print(myAtoi("words and 987"))