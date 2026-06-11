word = {}

inputword = input("Enter a word: ")


for i in inputword:
    if i in word:
        word[i] = word[i] + 1
    else:
        word[i] = 1 


for i in inputword:
    print(i,"出现了",word[i],"次")
