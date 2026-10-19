def isIsomorphic(s, t):
    if len(s) != len(t):
        return False
    s_to_t = {}
    t_to_s = {}
    for ch_s, ch_t in zip(s, t):
        if ch_s in s_to_t:
            if s_to_t[ch_s] != ch_t:
                return False
        else:
            s_to_t[ch_s] = ch_t
        if ch_t in t_to_s:
            if t_to_s[ch_t] != ch_s:
                return False
        else:
            t_to_s[ch_t] = ch_s
    return True

if __name__ == "__main__":
    print(isIsomorphic("egg", "add"))
    print(isIsomorphic("f11", "b23"))
    print(isIsomorphic("paper", "title"))