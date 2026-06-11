class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        if not prices or len(prices) < 2:
            return 0
        n = len(prices)
        left = [0] * n
        min_price = prices[0]
        for i in range(1, n):
            min_price = min(min_price, prices[i])
            left[i] = max(left[i-1], prices[i] - min_price)
        right = [0] * n
        max_price = prices[-1]
        for i in range(n-2, -1, -1):
            max_price = max(max_price, prices[i])
            right[i] = max(right[i+1], max_price - prices[i])
        ans = 0
        for i in range(n):
            ans = max(ans, left[i] + right[i])
        return ans


prices = [3,3,5,0,0,3,1,4]
print(Solution().maxProfit(prices))

prices = [1,2,3,4,5]
print(Solution().maxProfit(prices))

prices = [7,6,4,3,1]
print(Solution().maxProfit(prices))

prices = [1]
print(Solution().maxProfit(prices))
