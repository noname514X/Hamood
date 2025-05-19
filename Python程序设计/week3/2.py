print([p for p in range (2,100) if 0 not in [p%d for d in range(2, int(p**0.5) + 1)]])

matrix = [[1,2,3],[4,5,6],[7,8,9]]
print(matrix)
print([[row[i] for row in matrix] for i in range(3)])
print(list(zip(*matrix)))
print(list(zip(matrix[0],matrix[1],matrix[2])))
print(list(zip(matrix[0],matrix[1],matrix[2],matrix[0],matrix[1],matrix[2])))
print(list(zip(matrix[0],matrix[1],matrix[2],matrix[0],matrix[1],matrix[2],matrix[0],matrix[1],matrix[2])))
print(list(zip(matrix[0],matrix[1],matrix[2],matrix[0],matrix[1],matrix[2],matrix[0],matrix[1],matrix[2],matrix[0],matrix[1],matrix[2])))