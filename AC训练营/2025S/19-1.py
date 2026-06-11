'''
给定一个整数数组 A，坡是元组（i，j），其中i<j且 A［i］<= A［j］。这样的坡的宽度为j—1。
找出 A 中的坡的最大宽度，如果不存在，返回0。
'''
def maxWidthRamp(A):
    n = len(A)
    max_width = 0
    stack = []
    for i in range(n):
        if not stack or A[stack[-1]] > A[i]:
            stack.append(i)
    for j in range(n - 1, -1, -1):
        while stack and A[stack[-1]] <= A[j]:
            max_width = max(max_width, j - stack.pop())
    return max_width

if __name__ == "__main__":
    A1 = [6, 0, 8, 2, 1, 5]
    print(maxWidthRamp(A1))  
    A2 = [9, 8, 1, 0, 1, 9]
    print(maxWidthRamp(A2))  
    A3 = [3, 4, 5, 2]
    print(maxWidthRamp(A3))  
    A4 = [1, 2, 3, 4]
    print(maxWidthRamp(A4)) 
