import sys, re, ast
from collections import deque

def numIslands(grid):
    if not grid:
        return 0
    m,n=len(grid),len(grid[0])
    cnt=0
    for i in range(m):
        for j in range(n):
            if grid[i][j]=='1':
                cnt+=1
                dq=deque([(i,j)])
                grid[i][j]='0'
                while dq:
                    x,y=dq.popleft()
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<m and 0<=ny<n and grid[nx][ny]=='1':
                            grid[nx][ny]='0'
                            dq.append((nx,ny))
    return cnt

def main():
    data=sys.stdin.read()
    m=re.search(r"\[[\s\S]*\]",data)
    grid=[]
    if m:
        try:
            grid=ast.literal_eval(m.group(0))
        except Exception:
            grid=[]
    print(numIslands(grid))

if __name__=='__main__':
    main()
