'''
给定一个整数数组 temperatures，表示每天的温度，返回一个数组 answer，其中 answer［1］是指对于第1天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用0来代替。
'''

def dailyTemperatures(temperatures):
    n = len(temperatures)
    answer = [0] * n
    stack = []  

    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            idx = stack.pop()
            answer[idx] = i - idx
        stack.append(i)

    return answer

if __name__ == "__main__":
    temperatures1 = [73, 74, 75, 71, 69, 72, 76, 73]
    print(dailyTemperatures(temperatures1))
    temperatures2 = [30, 40, 50, 60]
    print(dailyTemperatures(temperatures2))
    temperatures3 = [30, 60, 90]
    print(dailyTemperatures(temperatures3))
