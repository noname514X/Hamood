'''
给你一个整数数组 nums。玩家1和玩家 2 基于这个数组设计了一个游戏。
玩家1和玩家2轮流进行自己的回合，玩家1先手。开始时，两个玩家的初始分值都是 0。每一回合，玩家从数组的任意一端取一个数字（即，nums［Q1 或 nums ［nums. Length - 1］），取到的数字将会从数组中移除（数组长度减 1）。玩家选中的数字将会加到他的得分上。当数组中没有剩余数字可取时，游戏结束。
如果玩家1能成为赢家，返回true。如果两个玩家得分相等，同样认为玩家1是游戏的赢家，也返回 true。你可以假设每个玩家的玩法都会使他的分数最大化。
'''


def PredictTheWinner(nums):
    n = len(nums)
    memo = {}
    def helper(left, right):
        if left == right:
            return nums[left]
        if (left, right) in memo:
            return memo[(left, right)]
        pick_left = nums[left] - helper(left + 1, right)
        pick_right = nums[right] - helper(left, right - 1)
        memo[(left, right)] = max(pick_left, pick_right)
        return memo[(left, right)]
    return helper(0, n - 1) >= 0


if __name__ == "__main__":
    nums1 = [1, 5, 2]
    print(PredictTheWinner(nums1)) 
    nums2 = [1, 5, 233, 7]
    print(PredictTheWinner(nums2)) 
    nums3 = [1, 1, 1, 1]
    print(PredictTheWinner(nums3)) 
    nums4 = [2]
    print(PredictTheWinner(nums4)) 
