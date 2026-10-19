def findContentChildren(g, s):
    g.sort()
    s.sort()
    i = 0
    for cookie in s:
        if i < len(g) and cookie >= g[i]:
            i += 1
    return i

if __name__ == "__main__":
    print(findContentChildren([1, 2, 3], [1, 1]))
    print(findContentChildren([1, 2], [1, 2, 3]))