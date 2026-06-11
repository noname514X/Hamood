'''
给定在 xy平面上的一组点，确定由这些点组成的矩形的最小面积，其中矩形的边平行于×轴和 y轴。
如果没有任何矩形，就返回0。
'''


def minAreaRect(points):
    point_set = set((x, y) for x, y in points)
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            if x1 != x2 and y1 != y2:
                if (x1, y2) in point_set and (x2, y1) in point_set:
                    area = abs(x1 - x2) * abs(y1 - y2)
                    if area < min_area:
                        min_area = area
    return 0 if min_area == float('inf') else min_area


if __name__ == "__main__":
    points1 = [[1,1],[1,3],[3,1],[3,3],[2,2]]
    print(minAreaRect(points1))  # 4
    points2 = [[1,1],[1,3],[3,1],[3,3]]
    print(minAreaRect(points2))  # 4
    points3 = [[1,1],[1,2],[1,3],[3,1],[3,2],[3,3]]
    print(minAreaRect(points3))  # 2
    points4 = [[0,1],[1,0],[2,1],[1,2]]
    print(minAreaRect(points4))  # 0
