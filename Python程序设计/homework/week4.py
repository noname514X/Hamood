#编写程序，输入一个字符串，输出其中出现次数最多的字符及其出现的次数。要求使用字典。
# from collections import Counter
# word = input("请输入一个字符串：")
# print(word)
# frequences = Counter(word)
# frequences.items()
# print(frequences.most_common(1))

word = input("请输入一个字符串：")

word_dict = {}
for i in word:
    if i in word_dict:
        word_dict[i] += 1
    else:
        word_dict[i] = 1

max_char = ''
max_count = 0
for k, v in word_dict.items():
    if v > max_count:
        max_char = k
        max_count = v

print(max_char)