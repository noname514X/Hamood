'''
如果长度为 n 的数组 nums 满足下述条件，则认为该数组是一个 漂亮数组：
•nums 是由范围［2，n］的整数组成的一个排列。
•对于每个 @<= i < j<n，均不存在下标 k（i< k < j）使得2 * nums ［k］ == nums ［i］ + nums ［j］。
给你整数n，返回长度为n 的任一漂亮数组。本题保证对于给定的 n/至少存在一个有效答案。
'''
def beautifulArray(n):
    def generate(n):
        if n == 1:
            return [1]
        odd = generate((n + 1) // 2)
        even = generate(n // 2)
        return [x * 2 - 1 for x in odd] + [x * 2 for x in even]

    return generate(n)

if __name__ == "__main__":
    n1 = 5
    print(beautifulArray(n1))
    n2 = 6
    print(beautifulArray(n2)) 
    n3 = 7
    print(beautifulArray(n3))  
    n4 = 8
    print(beautifulArray(n4)) 
