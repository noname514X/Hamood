def lengthOfLastWord(s):
    length = 0
    i = len(s) - 1
    while i >= 0 and s[i] == ' ':
        i -= 1
    while i >= 0 and s[i] != ' ':
        length += 1
        i -= 1
    return length

if __name__ == "__main__":
    print(lengthOfLastWord("Hello World"))
    print(lengthOfLastWord("fly me to the moon"))
    print(lengthOfLastWord("luffy is still joyboy"))