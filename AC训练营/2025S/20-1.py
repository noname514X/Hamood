'''
在歌曲列表中，第i首歌曲的持续时间为 time ［i秒。
返回其总持续时间（以秒为单位）可被 60整除的歌曲对的数量。形式上，我们希望下标数字i和j满足i< j且有（time［i］+ time［j］）& 60 ==0。
'''

def numPairsDivisibleBy60(time):
    count = [0] * 60
    ans = 0
    for t in time:
        mod = t % 60
        ans += count[(60 - mod) % 60]
        count[mod] += 1
    return ans


if __name__ == "__main__":
    time1 = [30,20,150,100,40]
    print(numPairsDivisibleBy60(time1))  # 3
    time2 = [60,60,60]
    print(numPairsDivisibleBy60(time2))  # 3
    time3 = [10,50,90,30]
    print(numPairsDivisibleBy60(time3))  # 2
    time4 = [1,2,3,4,5]
    print(numPairsDivisibleBy60(time4))  # 0

