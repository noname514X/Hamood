def removeElement(nums, val):
    k = 0
    for i in range(len(nums)):
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1
    return k

if __name__ == "__main__":
    nums1 = [3, 2, 2, 3]
    k1 = removeElement(nums1, 3)
    print(k1, nums1[:k1])
    nums2 = [0, 1, 2, 2, 3, 0, 4, 2]
    k2 = removeElement(nums2, 2)
    print(k2, nums2[:k2])