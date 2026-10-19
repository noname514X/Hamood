def convertToBase7(num):
    if num == 0:
        return "0"
    negative = num < 0
    num = abs(num)
    digits = []
    while num > 0:
        digits.append(str(num % 7))
        num //= 7
    if negative:
        digits.append('-')
    return ''.join(reversed(digits))

if __name__ == "__main__":
    print(convertToBase7(100))
    print(convertToBase7(-7))