def arrayPairSum(nums):
    nums.sort()
    return sum(nums[0::2])

if __name__ == "__main__":
    print(arrayPairSum([1, 4, 3, 2]))
    print(arrayPairSum([6, 2, 6, 5, 1, 2]))