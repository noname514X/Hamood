'''
给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations［i］的长度为 4，并采用两种不同的形式之一：“a==b”或“a！=0”。在这里，a和b是小写字母（不一定不同），表示单字母变量名。
只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，否则返回false。
'''


def equationsPossible(equations):
    parent = [i for i in range(26)]  # 26个小写字母
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        parent[find(x)] = find(y)

    for eq in equations:
        if eq[1:3] == '==':
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            union(x, y)

    for eq in equations:
        if eq[1:3] == '!=':
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            if find(x) == find(y):
                return False
    return True

# 测试用例
if __name__ == "__main__":
    eqs1 = ["a==b","b!=a"]
    print(equationsPossible(eqs1)) 
    eqs2 = ["b==a","a==b"]
    print(equationsPossible(eqs2)) 
    eqs3 = ["a==b","b==c","a==c"]
    print(equationsPossible(eqs3)) 
    eqs4 = ["a==b","b!=c","c==a"]
    print(equationsPossible(eqs4))  
