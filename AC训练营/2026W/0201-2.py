class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        if n <= 1:
            return 0
        pal = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                pal[i][j] = (s[i] == s[j]) and (j - i < 2 or pal[i + 1][j - 1])
        dp = [0] * n
        for i in range(n):
            if pal[0][i]:
                dp[i] = 0
            else:
                best = i
                for j in range(1, i + 1):
                    if pal[j][i]:
                        best = min(best, dp[j - 1] + 1)
                dp[i] = best
        return dp[-1]

line = input().strip()
if '=' in line:
    s = line.split('=', 1)[1].strip()
else:
    s = line
if len(s) >= 2 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
    s = s[1:-1]
print(Solution().minCut(s))
