'''
假设有从1到n的n个整数。用这些整数构造一个数组 perm（下标从 1开始），只要满足下述条件 之一，该数组就是一个优美的排列：
•perm［i］ 能够被1整除
•i能够被 perm［i］ 整除
给你一个整数n，返回可以构造的 优美排列 的数量。
'''

def countArrangement(n):
    def backtrack(pos, visited):
        if pos > n:
            return 1
        total = 0
        for i in range(1, n + 1):
            if not visited[i] and (i % pos == 0 or pos % i == 0):
                visited[i] = True
                total += backtrack(pos + 1, visited)
                visited[i] = False
        return total

    return backtrack(1, [False] * (n + 1))

if __name__ == "__main__":
    print(countArrangement(1))  # 1
    print(countArrangement(2))  # 2
    print(countArrangement(3))  # 3
    print(countArrangement(4))  # 8
