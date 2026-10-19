def moveZeroes(nums):
    pos = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[pos] = nums[i]
            pos += 1
    while pos < len(nums):
        nums[pos] = 0
        pos += 1

if __name__ == "__main__":
    nums1 = [0, 1, 0, 3, 12]
    moveZeroes(nums1)
    print(nums1)
    nums2 = [0]
    moveZeroes(nums2)
    print(nums2)