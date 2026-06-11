'''
给你一个大小为 n x n 的二元矩阵 grid，其中 1 表示陆地，@表示水域。
岛是由四面相连的 1形成的一个最大组，即不会与非组内的任何其他 1 相连。grid 中恰好存在两座岛。
你可以将任意数量的◎变为 1，以使两座岛连接起来，变成一座岛。
返回必须翻转的 ◎的最小数目。
'''
def shortestBridge(grid):
    from collections import deque

    n = len(grid)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def bfs(start):
        queue = deque([start])
        visited = set([start])
        island = []
        
        while queue:
            x, y = queue.popleft()
            island.append((x, y))
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in visited and grid[nx][ny] == 1:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return island
    

    found = False
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                first_island = bfs((i, j))
                found = True
                break
        if found:
            break
    

    queue = deque(first_island)
    distance = 0
    visited = set(first_island)
    
    while queue:
        for _ in range(len(queue)):
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    if (nx, ny) not in visited:
                        if grid[nx][ny] == 1:  # Found the second island
                            return distance
                        elif grid[nx][ny] == 0:  # Water cell
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        distance += 1
    
    return -1  # Should not reach here if there are exactly two islands

if __name__ == "__main__":
    grid1 = [[0, 1], [1, 0]]
    print(shortestBridge(grid1)) 

    grid2 = [[0, 1, 0], [0, 0, 0], [0, 0, 1]]
    print(shortestBridge(grid2))  

    grid3 = [[1, 1], [0, 1]]
    print(shortestBridge(grid3))  
