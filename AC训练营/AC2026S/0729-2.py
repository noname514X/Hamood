def countSegments(s):
    count = 0
    in_word = False
    for ch in s:
        if ch != ' ':
            if not in_word:
                count += 1
                in_word = True
        else:
            in_word = False
    return count

if __name__ == "__main__":
    print(countSegments("Hello, my name is John"))