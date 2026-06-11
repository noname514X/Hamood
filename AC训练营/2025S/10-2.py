'''
给你一个整数数组 nums 和两个整数：left 及 right。找出 nums 中连续、非空且其中最大元素在范围 ［left，right］内的子数组，并返回满足条件的子数组的个数。
生成的测试用例保证结果符合 32-bit 整数范围。
'''
def countSubarrays(nums, left, right):
    count = 0
    n = len(nums)
    for i in range(n):
        max_num = nums[i]
        for j in range(i, n):
            max_num = max(max_num, nums[j])
            if left <= max_num <= right:
                count += 1
            if max_num > right:
                break
    return count

if __name__ == "__main__":
    nums1 = [1, 2, 3, 4]
    left1 = 2
    right1 = 3
    print(countSubarrays(nums1, left1, right1))  

    nums2 = [5, 6, 7, 8]
    left2 = 5
    right2 = 7
    print(countSubarrays(nums2, left2, right2)) 

    nums3 = [1, 3, 5, 7]
    left3 = 4
    right3 = 6
    print(countSubarrays(nums3, left3, right3)) 
