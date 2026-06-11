'''
两个整数的 汉明距离 指的是这两个数字的二进制数对应位不同的数量。
给你一个整数数组 nums，请你计算并返回 nums中任意两个数之间 汉明距离的总和。
'''
def totalHammingDistance(nums):
    total = 0
    n = len(nums)
    for i in range(32):
        count = sum((num >> i) & 1 for num in nums)
        total += count * (n - count)
    return total

if __name__ == "__main__":
    nums1 = [4, 14, 2]
    print(totalHammingDistance(nums1))  
    nums2 = [4, 14, 4]
    print(totalHammingDistance(nums2))  
    nums3 = [2, 3]
    print(totalHammingDistance(nums3))  
    nums4 = [1, 2, 3]
    print(totalHammingDistance(nums4))  
