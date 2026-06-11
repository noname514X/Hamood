'''
给定一个由4位数字组成的数组，返回可以设置的符合 24小时制的最大时间。
24小时格式为 “HH:MM”，其中HH 在00到23之间，MM在00到59之间。最小的24小时制时间是00:00，而最大的是 23:59。从00:00（午夜）开始算起，过得越久，时间越大。
以长度为5的字符串，按“HH:MM”格式返回答案。如果不能确定有效时间，则返回空字符串。
'''

def largestTimeFromDigits(A):
    A.sort(reverse=True)
    for h1 in A:
        for h2 in A:
            if h2 == h1: continue
            for m1 in A:
                if m1 == h1 or m1 == h2: continue
                for m2 in A:
                    if m2 == h1 or m2 == h2 or m2 == m1: continue
                    hour = 10 * h1 + h2
                    minute = 10 * m1 + m2
                    if hour < 24 and minute < 60:
                        return f"{hour:02}:{minute:02}"
    return ""

if __name__ == "__main__":
    print(largestTimeFromDigits([1, 2, 3, 4]))  # "23:41"
    print(largestTimeFromDigits([5, 5, 5, 5]))  # ""
    print(largestTimeFromDigits([0, 0, 0, 0]))  # "00:00"
    print(largestTimeFromDigits([2, 3, 5, 9]))  # "23:50"
