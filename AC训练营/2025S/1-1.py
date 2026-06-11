'''
在LeetCode商店中，有n件在售的物品。每件物品都有对应的价格。然而，也有一些大礼包，每个大礼包以优惠的价格捆绑销售一组物品。
给你一个整数数组 price 表示物品价格，其中 price［i］是第i件物品的价格。另有一个整数数组 needs 表示购物清单，其中 needs ［i〕是需要购买第i件物品的数量。
还有一个数组 special表示大礼包，special［i］的长度为n+1，其中special ［i］ ［j］表示第1个大礼包中内含第j件物品的数量，且 special［1］［n］（也就是数组中的最后一个整数）为第i个大礼包的价格。
返回 确切 满足购物清单所需花费的最低价格，你可以充分利用大礼包的优惠活动。你不能购买超出购物清单指定数量的物品，即使那样会降低整体价格。任意大礼包可无限次购买。
'''

def shoppingOffers(price, special, needs):
    total = 0
    for i in range(len(price)):
        total += price[i] * needs[i]
    for offer in special:
        valid = True
        new_needs = []
        for i in range(len(needs)):
            if needs[i] < offer[i]:
                valid = False
                break
            new_needs.append(needs[i] - offer[i])
        if valid:
            total = min(total, shoppingOffers(price, special, new_needs) + offer[-1])
    return total


if __name__ == "__main__":
    price = [2, 5]
    special = [[3, 0, 5], [1, 2, 10]]
    needs = [3, 2]
    print(shoppingOffers(price, special, needs))

