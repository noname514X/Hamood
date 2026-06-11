class Solution:
    def partition(self,s: str):
        n = len(s)
        pal = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                pal[i][j] = (s[i] == s[j]) and (j - i < 2 or pal[i + 1][j - 1])
        res = []
        path = []
        def dfs(start):
            if start == n:
                res.append(path.copy())
                return
            for end in range(start, n):
                if pal[start][end]:
                    path.append(s[start:end + 1])
                    dfs(end + 1)
                    path.pop()
        dfs(0)
        return res

line = input().strip()
if '=' in line:
    s = line.split('=', 1)[1].strip()
else:
    s = line
if len(s) >= 2 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
    s = s[1:-1]
print(Solution().partition(s))
