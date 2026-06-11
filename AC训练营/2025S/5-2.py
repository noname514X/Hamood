'''
给出一个字符串数组 words 组成的一本英语词典。返回能够通过 words中其它单词逐步添加一个字母来构造得到的 words 中最长的单词。
若其中有多个可行的答案，则返回答案中字典序最小的单词。若无答案，则返回空字符串。
请注意，单词应该从左到右构建，每个额外的字符都添加到前一个单词的结尾。
'''

from typing import List
def longestWord(words: List[str]) -> str:
    words_set = set(words)
    longest_word = ""
    
    for word in words:
        if len(word) > len(longest_word) or (len(word) == len(longest_word) and word < longest_word):
           
            if all(word[:i] in words_set for i in range(1, len(word))):
                longest_word = word
                
    return longest_word

if __name__ == "__main__":
    words = ["w", "wo", "wor", "worl", "world"]
    print(longestWord(words))  
    
    words = ["a", "banana", "app", "appl", "ap"]
    print(longestWord(words))  
    
    words = ["a", "b", "ba"]
    print(longestWord(words))  
