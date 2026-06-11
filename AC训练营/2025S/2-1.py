'''
你将得到一个整数数组 matchsticks，其中matchsticks ［i］是第1个火柴棒的长度。你要用所有的火紫棍 拼成一个正方形。你 不能折断任何一根火柴棒，但你可以把它们连在一起，而且每根火柴棒必须使用一次。
如果你能使这个正方形，则返回 true，否则返回false。
'''


def makesquare(matchsticks):
    if not matchsticks or len(matchsticks) < 4:
        return False
    total = sum(matchsticks)
    if total % 4 != 0:
        return False
    side = total // 4
    matchsticks.sort(reverse=True) 
    sides = [0] * 4

    def dfs(index):
        if index == len(matchsticks):
            return sides[0] == sides[1] == sides[2] == sides[3] == side
        for i in range(4):
            if sides[i] + matchsticks[index] <= side:
                sides[i] += matchsticks[index]
                if dfs(index + 1):
                    return True
                sides[i] -= matchsticks[index]
        return False

    return dfs(0)


if __name__ == "__main__":

    test1 = [1,1,2,2,2]
    print(makesquare(test1)) 
    
    test2 = [3,3,3,3,4]
    print(makesquare(test2)) 

    test3 = [1,1,1,2,2,2,3,3,4,4]
    print(makesquare(test3)) 

    test4 = [0,0,0,0]
    print(makesquare(test4)) 
