def demo(s):
    result = [0,0,0,0]
    for ch in s:
        if 'a' <= ch <= 'z':
            result[1] += 1
        elif 'A' <= ch <= 'Z':
            result[0] += 1
        elif '0' <= ch <= '9':
            result[2] += 1
        else:
            result[3] += 1
    return tuple(result)

print(demo('iudyiIYSIuadhHBASu01380498021)_(S*)_&*(ASD()&)AS%DK@LQe,..013bdnasl..,,,.db'))