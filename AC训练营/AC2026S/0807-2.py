def isPowerOfTwo(n):
    return n > 0 and (n & (n - 1)) == 0

if __name__ == "__main__":
    print(isPowerOfTwo(1))
    print(isPowerOfTwo(16))
    print(isPowerOfTwo(3))
    print(isPowerOfTwo(0))