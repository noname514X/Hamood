'''
给你一个按递增顺序排序的数组 arr 和一个整数k。数组 arr 由1 和若干 质数 组成，且其中所有整数互不相同。
对于每对满足 0 <= i < j <arr.length 的i和j，可以得到分数 arr［i］ / arr［j］。
那么第 k 个最小的分数是多少呢？以长度为2 的整数数组返回你的答案，这里 answer［0］== arr［i］ 且 answer［1］ == arr［j］。
'''
def kthSmallestPrimeFraction(arr, k):
    from fractions import Fraction
    fractions = [Fraction(arr[i], arr[j]) for i in range(len(arr)) for j in range(i + 1, len(arr))]
    fractions.sort()
    return [fractions[k - 1].numerator, fractions[k - 1].denominator]
if __name__ == "__main__":
    arr1 = [1, 2, 3, 5]
    k1 = 3
    print(kthSmallestPrimeFraction(arr1, k1))  
    
    arr2 = [1, 2, 3, 4]
    k2 = 2
    print(kthSmallestPrimeFraction(arr2, k2))  
