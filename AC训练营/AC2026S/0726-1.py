def addStrings(num1, num2):
    i, j = len(num1) - 1, len(num2) - 1
    carry = 0
    result = []
    while i >= 0 or j >= 0 or carry:
        n1 = ord(num1[i]) - ord('0') if i >= 0 else 0
        n2 = ord(num2[j]) - ord('0') if j >= 0 else 0
        total = n1 + n2 + carry
        carry = total // 10
        result.append(str(total % 10))
        i -= 1
        j -= 1
    return ''.join(reversed(result))

if __name__ == "__main__":
    print(addStrings("11", "123"))
    print(addStrings("456", "77"))
    print(addStrings("0", "0"))