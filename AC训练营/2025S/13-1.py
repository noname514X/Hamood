'''
Alice 和Bob用几堆石子在做游戏。一共有偶数堆石子，排成一行；每堆都有 正 整数颗石子，数目为 piles［。
游戏以谁手中的石子最多来决出胜负。石子的 总数 是 奇数，所以没有平局。
Alice 和 Bob 轮流进行，Alice 先开始。每回合，玩家从行的 开始 或结束 处取走整堆石头。这种情况一直持续到没有更多的石子堆为止，此时手中 石子最多 的玩家 获胜。
假设 Alice 和 Bob 都发挥出最佳水平，当 Alice 赢得比赛时返回 true，当 Bob 赢得比赛时返回 false
'''

def stoneGame(piles):
    n = len(piles)
    memo = {}
    def helper(left, right):
        if left == right:
            return piles[left]
        if (left, right) in memo:
            return memo[(left, right)]
        pick_left = piles[left] - helper(left + 1, right)
        pick_right = piles[right] - helper(left, right - 1)
        memo[(left, right)] = max(pick_left, pick_right)
        return memo[(left, right)]
    return helper(0, n - 1) > 0


if __name__ == "__main__":
    piles1 = [5,3,4,5]
    print(stoneGame(piles1))  
    piles2 = [3,7,2,3]
    print(stoneGame(piles2)) 
    piles3 = [1,100,2]
    print(stoneGame(piles3)) 
    piles4 = [1,2]
    print(stoneGame(piles4)) 
    
