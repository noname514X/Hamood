'''
你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字：“@”，‘2”，‘2”，'3'，'4”，“5'，96”，'7”，'8'，“9”。每个拨轮可以自由旋转：例如把•9'变为‘0'，‘0'变为
'9’。每次旋转都只能旋转一个拨轮的一位数字。
锁的初始数字为'0000'
，一个代表四个拨轮的数字的字符串。
列表 deadends包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。
字符串 target 代表可以解锁的数字，你需要给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回-1。
'''

from collections import deque

def openLock(deadends, target):
    dead = set(deadends)
    if '0000' in dead:
        return -1
    queue = deque()
    queue.append(('0000', 0))
    visited = set('0000')
    while queue:
        status, step = queue.popleft()
        if status == target:
            return step
        for i in range(4):
            for d in [-1, 1]:
                num = int(status[i])
                new_num = (num + d) % 10
                new_status = status[:i] + str(new_num) + status[i+1:]
                if new_status not in dead and new_status not in visited:
                    queue.append((new_status, step + 1))
                    visited.add(new_status)
    return -1


if __name__ == "__main__":

    deadends1 = ["0201","0101","0102","1212","2002"]
    target1 = "0202"
    deadends2 = ["8888"]
    target2 = "0009"
    print(openLock(deadends2, target2)) 
    deadends3 = ["0000"]
    target3 = "8888"
    print(openLock(deadends3, target3)) 
    deadends4 = []
    target4 = "0000"
    print(openLock(deadends4, target4)) 

