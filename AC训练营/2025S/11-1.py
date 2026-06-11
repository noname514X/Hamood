'''
在桌子上有 n 张卡片，每张卡片的正面和背面都写着一个正效（正面与背面上的数有可能不一样）。
我们可以先翻转任意张卡片，然后选择其中一张卡片。
如果选中的那张卡片背面的数字 ×与任意一张卡片的正面的数字都不同，那么这个数字是我们想要的数字。
哪个数是这些想要的数字中最小的数（找到这些数中的最小值）呢？如果没有一个数字符合要求的，输出。。
其中，fronts ［i］ 和 backs ［i］分别代表第i 张卡片的正面和背面的数字。
如果我们通过翻转卡片来交换正面与背面上的数，那么当初在正面的数就变成背面的数，背面的数就变成正面的数。
'''
def flipgame(fronts, backs):
    bad_numbers = set()
    for i in range(len(fronts)):
        if fronts[i] == backs[i]:
            bad_numbers.add(fronts[i])
    
    min_good_number = float('inf')
    for num in fronts + backs:
        if num not in bad_numbers:
            min_good_number = min(min_good_number, num)
    
    return min_good_number if min_good_number != float('inf') else 0

if __name__ == "__main__":
    fronts1 = [1, 2, 4, 4, 7]
    backs1 = [1, 3, 4, 1, 3]
    print(flipgame(fronts1, backs1)) 

    fronts2 = [1, 2]
    backs2 = [2, 1]
    print(flipgame(fronts2, backs2))  
    fronts3 = [1, 2, 3]
    backs3 = [4, 5, 6]
    print(flipgame(fronts3, backs3)) 
