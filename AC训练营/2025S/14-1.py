'''
给你两个字符串数组 words1 和 words2。
现在，如果 b中的每个字母都出现在a中，包括重复出现的字母，那么称字符串 b是字符串a的子集。
• 例如，“wrr"是“warrior”的子集，但不是“world™”的子集。
如果对 words2 中的每一个单词 b，b都是a的子集，那么我们称 words1 中的单词。是通用单词。
以数组形式返回 words1中所有的通用 单词。你可以按 任意顺序 返回答案。
'''

def wordSubsets(words1, words2):
    def countChars(word):
        count = [0] * 26
        for c in word:
            count[ord(c) - ord('a')] += 1
        return count

    maxCounts = [0] * 26
    for word in words2:
        currCounts = countChars(word)
        for i in range(26):
            maxCounts[i] = max(maxCounts[i], currCounts[i])

    result = []
    for word in words1:
        currCounts = countChars(word)
        if all(currCounts[i] >= maxCounts[i] for i in range(26)):
            result.append(word)

    return result

if __name__ == "__main__":
    words1 = ["amazon", "apple", "facebook", "google", "leetcode"]
    words2 = ["e", "o"]
    print(wordSubsets(words1, words2)) 
    words1 = ["amazon", "apple", "facebook", "google", "leetcode"]
    words2 = ["l", "e"]
    print(wordSubsets(words1, words2)) 
