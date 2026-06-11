'''
给你一个整数数组 nums，你可以对它进行一些操作。
每次操作中，选择任意一个 hums ［1〕，删除它并获得 nums［L］的点数。之后，你必须删除所有等于Tums ［1］-17和nums ［4］ +1的元素。
开始你拥有 ◎个点数。返回你能通过这些操作获得的最大点数。
'''
def maxPoints(nums):
    from collections import Counter
    count = Counter(nums)
    max_num = max(nums)
    dp = [0] * (max_num + 1)

    for i in range(1, max_num + 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + i * count[i])

    return dp[max_num]

if __name__ == "__main__":
    nums1 = [3, 4, 2]
    print(maxPoints(nums1))  
    nums2 = [2, 2, 3, 3, 3, 4]
    print(maxPoints(nums2))  
    nums3 = [1, 2, 3, 4, 5]
    print(maxPoints(nums3))  
    nums4 = [5, 5, 5, 5]
    print(maxPoints(nums4))  
