def findDisappearedNumbers(nums):
    n = len(nums)
    for num in nums:
        idx = abs(num) - 1
        if nums[idx] > 0:
            nums[idx] = -nums[idx]
    return [i + 1 for i in range(n) if nums[i] > 0]

if __name__ == "__main__":
    print(findDisappearedNumbers([4, 3, 2, 7, 8, 2, 3, 1]))
    print(findDisappearedNumbers([1, 1]))