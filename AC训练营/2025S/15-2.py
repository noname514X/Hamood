'''
给定一个整数数组 arr，以及一个整数 target 作为目标值，返回满足i<j<K 且 arr［i］ + arr［j］ + arr［k］ == target 的元组 i，j，k的数量。
由于结果会非常大，请返回 10°+7的模。
'''
def threeSumMulti(arr, target):
    from collections import Counter
    count = Counter(arr)
    result = 0
    mod = 10**9 + 7

    for i in range(len(arr)):
        count[arr[i]] -= 1
        for j in range(i + 1, len(arr)):
            count[arr[j]] -= 1
            complement = target - arr[i] - arr[j]
            if complement in count and count[complement] > 0:
                result += count[complement]
                result %= mod
            count[arr[j]] += 1
        count[arr[i]] += 1

    return result

if __name__ == "__main__":
    arr1 = [1, 2, 3, 4, 5]
    target1 = 8
    print(threeSumMulti(arr1, target1)) 

    arr2 = [1, 1, 2, 2, 3]
    target2 = 6
    print(threeSumMulti(arr2, target2))  

    arr3 = [0, 0, 0]
    target3 = 0
    print(threeSumMulti(arr3, target3))  
