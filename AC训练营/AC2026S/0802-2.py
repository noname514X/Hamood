def plusOne(digits):
    carry = 1
    for i in range(len(digits) - 1, -1, -1):
        total = digits[i] + carry
        digits[i] = total % 10
        carry = total // 10
        if carry == 0:
            return digits
    if carry:
        return [1] + digits
    return digits

if __name__ == "__main__":
    print(plusOne([1, 2, 3]))
    print(plusOne([4, 3, 2, 1]))
    print(plusOne([9]))