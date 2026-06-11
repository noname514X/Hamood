'''
Alice 手中有一把牌，她想要重新排列这些牌，分成若干组，使每一组的牌数都是 groupSize ，并且由groupsize 张连续的牌组成。
给你一个整数数组 hand 其中 hand ［i］ 是写在第i张牌上的数值。如果她可能重新排列这些牌，返回 true ；否则，返回 false。
'''
def isNStraightHand(hand, groupSize):
    if len(hand) % groupSize != 0:
        return False
    from collections import Counter
    count = Counter(hand)
    for num in sorted(count):
        if count[num] > 0:
            for i in range(num, num + groupSize):
                if count[i] < count[num]:
                    return False
                count[i] -= count[num]
    return True

if __name__ == "__main__":
    hand1 = [1, 2, 3, 4, 5]
    groupSize1 = 4
    print(isNStraightHand(hand1, groupSize1))  

    hand2 = [1, 2, 3, 4, 5]
    groupSize2 = 3
    print(isNStraightHand(hand2, groupSize2)) 

    hand3 = [1, 2, 3, 4, 5, 6]
    groupSize3 = 2
    print(isNStraightHand(hand3, groupSize3))  
