'''
给定一个整数数组 prices，其中prices ［1］表示第 天的股票价格；墊数 fee 代表了交易股票的手续费用。
你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
返回获得利润的最大值。
注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
'''
def maxProfit(prices, fee):
    n = len(prices)
    if n == 0:
        return 0
    
    hold = -prices[0] 
    cash = 0           
    
    for i in range(1, n):
        hold = max(hold, cash - prices[i])  
        cash = max(cash, hold + prices[i] - fee)

    return cash

if __name__ == "__main__":
    prices1 = [1, 3, 2, 8, 4, 9]
    fee1 = 2
    print(maxProfit(prices1, fee1)) 

    prices2 = [1, 3, 7, 5, 10, 3]
    fee2 = 3
    print(maxProfit(prices2, fee2)) 

    prices3 = [5, 4, 3, 2, 1]
    fee3 = 1
    print(maxProfit(prices3, fee3))  
