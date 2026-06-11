'''
假设有从1到n的n个整数。用这些整数构造一个数组perm（下标从1开始），只要满足下述条件 之一，该数组就是一个 优美的排列：
• perm［i］能够被i整除
•i能够被 perm［i］整除
'''

def countArrangement(n):
    used = [False] * (n + 1) 
    ans = [0]  

    def backtrack(pos):
        if pos > n:
            ans[0] += 1
            return
        for num in range(1, n + 1):
            if not used[num]:
                if num % pos == 0 or pos % num == 0:
                    used[num] = True
                    backtrack(pos + 1)
                    used[num] = False

    backtrack(1)
    return ans[0]


if __name__ == "__main__":
    n = 2
    print(countArrangement(n))  

