'''
车上最初有 capacity 个空座位。车 只能 向一个方向行驶（也就是说，不允许掉头或改变方向）
给定整数 capacity 和一个数组 trips,trip［i］ = ［numPassengerss, from,tos］表示第i次旅行有 numPassengers：乘客，接他们和放他们的位置分别是 from和 toz。这些位置是从汽车的初始位置向东的公里数。
当且仅当你可以在所有给定的行程中接送所有乘客时，返回true，否则请返回false。
'''

def carPooling(trips, capacity):
    timeline = [0] * 1001 
    for numPassengers, fromPos, toPos in trips:
        timeline[fromPos] += numPassengers
        timeline[toPos] -= numPassengers

    current_passengers = 0
    for i in range(1001):
        current_passengers += timeline[i]
        if current_passengers > capacity:
            return False
    return True

if __name__ == "__main__":
    trips1 = [[2, 1, 5], [3, 3, 7]]
    capacity1 = 4
    print(carPooling(trips1, capacity1)) 

    trips2 = [[2, 1, 5], [3, 5, 7]]
    capacity2 = 5
    print(carPooling(trips2, capacity2)) 

