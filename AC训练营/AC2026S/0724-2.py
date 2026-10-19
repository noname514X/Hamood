def wordPattern(pattern, s):
    words = s.split()
    if len(pattern) != len(words):
        return False
    p_to_w = {}
    w_to_p = {}
    for p, w in zip(pattern, words):
        if p in p_to_w:
            if p_to_w[p] != w:
                return False
        else:
            p_to_w[p] = w
        if w in w_to_p:
            if w_to_p[w] != p:
                return False
        else:
            w_to_p[w] = p
    return True

if __name__ == "__main__":
    print(wordPattern("abba", "dog cat cat dog"))
    print(wordPattern("abba", "dog cat cat fish"))
    print(wordPattern("aaaa", "dog cat cat dog"))