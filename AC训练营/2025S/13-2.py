'''
如果序列 x1，x2，•，x2 满足下列条件，就说它是 斐波那契式的：
• n >= 3|
• 对于所有i+2<=n，都有 xi + Xi+1 == X1-2
给定一个 严格递增 的正整数数组形成序列 arr，找到 arr 中最长的斐波那契式的子序列的长度。如果不存在，返回0。
子序列 是通过从另一个序列 arr中删除任意数量的元素（包括删除0个元素）得到的，同时不改变剩余元素顺序。例如，［3，5，81是［3，4，5， 6，7，8］的子序列。
'''
def lenLongestFibSubseq(arr):
    n = len(arr)
    index = {x: i for i, x in enumerate(arr)}
    dp = [[2] * n for _ in range(n)]
    max_length = 0

    for j in range(n):
        for i in range(j):
            k = index.get(arr[j] - arr[i], -1)
            if k != -1 and k < i:
                dp[i][j] = dp[k][i] + 1
                max_length = max(max_length, dp[i][j])
    return max_length if max_length >= 3 else 0
if __name__ == "__main__":  
    arr1 = [1, 2, 3, 4, 5, 6, 7, 8]
    print(lenLongestFibSubseq(arr1))  
    
    arr2 = [1, 3, 7, 11, 12, 14, 18]
    print(lenLongestFibSubseq(arr2))  
    
    arr3 = [1, 2, 3]
    print(lenLongestFibSubseq(arr3))  
    arr4 = [1]
    print(lenLongestFibSubseq(arr4))   
