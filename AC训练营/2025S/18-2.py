'''
给定一个长度为偶数的整数数组 arr，只有对 arr 进行重组后可以满足“对于每个 0<=i < len（arr） / 2，都有 arr［2 *i+ 1］= 2*arr［2 *订”时，返回 true；否则，返回 false。
'''
def canReorderDoubled(arr):
    from collections import Counter
    count = Counter(arr)
    for x in sorted(count, key=abs):
        if count[x] > count[2 * x]:
            return False
        count[2 * x] -= count[x]
    return True

if __name__ == "__main__":
    arr1 = [3, 1, 3, 6]
    print(canReorderDoubled(arr1)) 
    arr2 = [2, 1, 2, 6]
    print(canReorderDoubled(arr2)) 
    arr3 = [4, -2, 2, -4]
    print(canReorderDoubled(arr3)) 
    arr4 = [1, 2, 4, 16]
    print(canReorderDoubled(arr4)) 
