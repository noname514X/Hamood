'''
冬季已经来临。你的任务是设计一个有固定加热半径的供暖器向所有房屋供暖。
在加热器的加热半径范围内的每个房屋都可以获得供暖。
现在，给出位于一条水平线上的房屋 houses 和供暖器 heaters 的位置，请你找出并返回可以覆盖所有房屋的最小加热半径。
注意：所有供暖語 hneaters 都遵循你的半径标准，加热的半径也一样。
'''


def findRadius(houses, heaters):
    houses.sort()
    heaters.sort()
    ans = 0
    for house in houses:
        min_dist = float('inf')
        for heater in heaters:
            min_dist = min(min_dist, abs(house - heater))
        ans = max(ans, min_dist)
    return ans


if __name__ == "__main__":
    houses1 = [1,2,3]
    heaters1 = [2]
    print(findRadius(houses1, heaters1)) 
    houses2 = [1,2,3,4]
    heaters2 = [1,4]
    print(findRadius(houses2, heaters2)) 
    houses3 = [1,5]
    heaters3 = [2]
    print(findRadius(houses3, heaters3))  
    houses4 = [1,2,3]
    heaters4 = [1,2,3]
    print(findRadius(houses4, heaters4)) 
