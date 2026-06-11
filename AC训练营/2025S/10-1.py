'''
有时候人们会用重复写一些字母来表示额外的感受，比如“hello” ->“heeello0o”，“hi" ->“hiii"。我们将相邻字母都相同的一串字符定义为相同字母组，例如：“h”，"eee”，，"ooo”。
对于一个给定的字符串S，如果另一个单词能够通过将一些字母组扩张从而使其和S相同，我们将这个单词定义为可扩张的（stretchy）。扩张操作定义如下：选择一个字母组（包含字母 c），然后往其中添加相同的字母。使其长度达到3或以上。
例如，以“hello”为例，我们可以对字母组“o”扩张得到“hellooo”，但是无法以同样的方法得到“helloo”因为字母组“0o”长度小于3。此外，我们可以进行另一种扩张“I->“II”以获得“hellllooo。如果s- “heLLLLLooo”，那么查询词“hello”是可扩张的，因为可以对它执行这两种扩张操作使得query -“hello" ->“hellooo” ->"hell111000"-s。
输入一组查询单词，输出其中可扩张的单词数量。
'''

def is_stretchy(s, query):
    i, j = 0, 0
    while i < len(s) and j < len(query):
        if s[i] == query[j]:
            i += 1
            j += 1
        elif i > 0 and s[i] == s[i - 1]:
            i += 1
        else:
            return False
    while i < len(s):
        if s[i] == s[i - 1]:
            i += 1
        else:
            return False
    return j == len(query)

def count_stretchy_words(S, queries):
    count = 0
    for query in queries:
        if is_stretchy(S, query):
            count += 1
    return count

if __name__ == "__main__":
    S = "heeello"
    queries = ["hello", "hi", "helo"]
    print(count_stretchy_words(S, queries))  
    queries2 = ["heeeellooo", "hii", "heeeelllooo"]
    print(count_stretchy_words(S, queries2))  
