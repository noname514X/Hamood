def hammingWeight(n):
    return bin(n).count('1')

if __name__ == "__main__":
    print(hammingWeight(11))
    print(hammingWeight(128))
    print(hammingWeight(2147483645))