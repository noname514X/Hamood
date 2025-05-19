#假设一段楼梯共15个台阶，小明一步最多能上3个台阶。编写程序计算小明上这段楼梯一共有多少种方法。

def stair(n):
    #递归法
    if n == 1:
        return 1
    elif n == 2:
        return 2
    elif n == 3:
        return 4
    else:
        return stair(n-1) + stair(n-2) + stair(n-3)
    
print(stair(15))

def stair2(n):
    #递推法
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    dp[3] = 4
    for i in range(4, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
    return dp[n]

print(stair2(15))