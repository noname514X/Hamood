'''
给你两个整数数组 persons 和 times。在选举中，第i张票是在时刻为 times［i］时投给候选人 persons［i］的。
对于发生在时刻t的每个查询，需要找出在t时刻在选举中领先的候选人的编号。
在t时刻投出的选票也将被计入我们的查询之中。在平局的情况下，最近获得投票的候选人将会获胜。
实现 TopVotedCandidate 类：
• TopVotedCandidate（int ［］ persons, int ［］ times）使用 persons 和 times 数组初始化对象。
• int qlint t）根据前而描述的规则，返回在时刻 t在选举中领先的候选人的编号。
'''
from collections import defaultdict
class TopVotedCandidate:

    def __init__(self, persons, times):
        self.times = times
        self.leader = []
        count = defaultdict(int)
        current_leader = -1
        max_votes = 0
        
        for i, person in enumerate(persons):
            count[person] += 1
            if count[person] >= max_votes:
                max_votes = count[person]
                current_leader = person
            self.leader.append(current_leader)
    def query(self, t):

        left, right = 0, len(self.times) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.times[mid] <= t:
                left = mid + 1
            else:
                right = mid - 1
        return self.leader[right] if right >= 0 else -1
    
if __name__ == "__main__":
    persons = [0, 1, 0, 1, 0, 1, 0]
    times = [0, 5, 10, 15, 20, 25, 30]
    topVotedCandidate = TopVotedCandidate(persons, times)
    print(topVotedCandidate.query(3))   
    print(topVotedCandidate.query(12)) 
    print(topVotedCandidate.query(25)) 
    print(topVotedCandidate.query(15)) 
    print(topVotedCandidate.query(24))  
    print(topVotedCandidate.query(8))   
