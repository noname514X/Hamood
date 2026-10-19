def addBinary(a, b):
    i, j = len(a) - 1, len(b) - 1
    carry = 0
    result = []
    while i >= 0 or j >= 0 or carry:
        bit_a = ord(a[i]) - ord('0') if i >= 0 else 0
        bit_b = ord(b[j]) - ord('0') if j >= 0 else 0
        total = bit_a + bit_b + carry
        carry = total // 2
        result.append(str(total % 2))
        i -= 1
        j -= 1
    return ''.join(reversed(result))

if __name__ == "__main__":
    print(addBinary("11", "1"))
    print(addBinary("1010", "1011"))