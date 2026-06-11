'''
给你一个整数数组 nums。每次 move 操作将会选择任意一个满足 @<= i < nums. length 的下标 i，并将 nums ［i］ 递增1。
返回使 nums 中的每个值都变成唯一的所需要的最少操作次数。
生成的测试用例保证答案在 32 位整数范围内。
'''
def minIncrementForUnique(nums):
    nums.sort()
    moves = 0
    next_unique = 0
    
    for num in nums:
        if num < next_unique:
            moves += next_unique - num
        next_unique = max(next_unique, num) + 1
    
    return moves

if __name__ == "__main__":
    nums1 = [1, 2, 2]
    print(minIncrementForUnique(nums1))  
    
    nums2 = [3, 2, 1, 2, 1, 7]
    print(minIncrementForUnique(nums2)) 
    
    nums3 = [1, 0, 2, 0]
    print(minIncrementForUnique(nums3)) 
    
    nums4 = [0, 0, 0]
    print(minIncrementForUnique(nums4))  
