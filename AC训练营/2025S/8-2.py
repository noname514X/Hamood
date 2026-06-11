'''
森林中有未知数量的兔子。提问其中若干只兔子”还有多少只兔子与你（指被提问的兔子）颜色相同？”，将答案收集到一个整数数组 answers中，其中 answers ［i］是第1只兔子的回答。
给你数组
answers，返回森林中兔子的最少数量。
'''
def numRabbits(answers):
    from collections import Counter
    count = Counter(answers)
    total = 0
    for k, v in count.items():
        total += (v + k) // (k + 1) * (k + 1)
    return total

if __name__ == "__main__":
    answers1 = [1, 1, 2]
    print(numRabbits(answers1)) 
    answers2 = [10, 10, 10]
    print(numRabbits(answers2))  
