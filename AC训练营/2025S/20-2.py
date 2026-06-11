'''
给你一个整数数组 nums，返回 nums 中最长等差子序列的长度。
回想一下，nums的子序列是一个列表 nums ［iu］，nums ［iz］，...nums［ixl，且0<=i<iz< .. <ik <= nuns.length - 1。并且如果 seq［i+1］ - seq［i］（e <= i < sea.length - 1）的值都相同，那么序列 seq 是等差的。
'''
def longestArithSeqLength(nums):
    if not nums:
        return 0
    
    n = len(nums)
    dp = [{} for _ in range(n)]
    max_length = 0
    for j in range(n):
        for i in range(j):
            diff = nums[j] - nums[i]
            if diff in dp[i]:
                dp[j][diff] = dp[i][diff] + 1
            else:
                dp[j][diff] = 2
            max_length = max(max_length, dp[j][diff])
    return max_length

if __name__ == "__main__":
    nums1 = [3, 6, 9, 12]
    print(longestArithSeqLength(nums1))  
    
    nums2 = [9, 4, 7, 2, 10]
    print(longestArithSeqLength(nums2)) 

    nums3 = [20, 1, 15, 3, 10, 5]
    print(longestArithSeqLength(nums3)) 
    
    nums4 = [1]
    print(longestArithSeqLength(nums4))  
