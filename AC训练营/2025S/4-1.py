'''
给定一个 24小时制（小时：分钟“HH:MM"）的时间列表，找出列表中任意两个时间的最小时间差并以分钟数表示。
'''

def findMinDifference(timePoints):
    minutes = []
    for t in timePoints:
        h, m = map(int, t.split(':'))
        minutes.append(h * 60 + m)
    minutes.sort()
    min_diff = 24 * 60
    for i in range(1, len(minutes)):
        min_diff = min(min_diff, minutes[i] - minutes[i-1])

    min_diff = min(min_diff, 24*60 - (minutes[-1] - minutes[0]))
    return min_diff


if __name__ == "__main__":
    times1 = ["23:59", "00:00"]
    print(findMinDifference(times1))  # 1
    times2 = ["01:01", "02:01", "03:00"]
    print(findMinDifference(times2))  # 59
    times3 = ["05:31", "22:08", "00:35"]
    print(findMinDifference(times3))  # 147
    times4 = ["00:00", "12:00", "06:00"]
    print(findMinDifference(times4))  # 360
