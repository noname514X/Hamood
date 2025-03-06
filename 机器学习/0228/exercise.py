#23089056

#p35页习题
def p35():
    import numpy as np
    def minkowski_distance(a, b, p):
        diff = np.abs(a - b) ** p
        distance = np.sum(diff) ** (1 / p)
        return distance

    def cosine_distance(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cosine_similarity = dot_product / (norm_a * norm_b)
        cosine_distance = 1 - cosine_similarity
        return cosine_distance

    vector1 = np.array([5,3])
    vector2 = np.array([1,4])

    distance = np.linalg.norm(vector1 - vector2)
    manhattan_distance = np.linalg.norm(vector1 - vector2, ord=1)
    minkowski_distance = minkowski_distance(vector1, vector2, 2)
    cosine_distance = cosine_distance(vector1, vector2)

    print('向量1:',vector1)
    print('向量2:',vector2)
    print()
    print('欧氏距离：',distance)
    print('曼哈顿距离',manhattan_distance)
    print('闵可夫斯基距离',manhattan_distance)
    print('余弦距离',cosine_distance)

#p36页习题
def p36():
    import numpy as np
    #36.1
    print("p36页第1题：生成数组元素是0至9的一维数组MyArray。")
    MyArray = np.arange(0,10,1)
    print()
    #36.2
    print("p36页第2题：对MyArray数组元素分别计算均值、标准差、总和、最大值。")
    mean = np.mean(MyArray)
    sum = np.sum(MyArray)
    std = np.std(MyArray)
    max = np.max(MyArray)
    print("均值：",mean)
    print("总和：",sum)
    print("标准差：",std)
    print("最大值：",max)
    print()
    #36.3
    print("p36页第3题：利用数组方法.cumsum()计算MyArray数组元素的当前累计和。")
    cumsum = np.cumsum(MyArray)
    print("累加和：",cumsum)
    print()
    #36.4
    print("p36页第4题：利用NumPy函数sqrt()对数组元素开平方。")
    sqrt = np.sqrt(MyArray)
    print("开方：",sqrt)
    print()
    #36.5
    print("p36页第5题：利用NumPy函数seed()指定随机数种子。")
    np.random.seed(1)
    randomNumber = np.random.randn(3,3)
    print("随机数：\n",randomNumber)
    print()
    #36.6
    print("p36页第6题：利用NumPy函数random.randn()生成包含10个元素且服从标准正态分布的一维数组。")
    randomNormalArray = np.random.normal(0,1,10)
    print("随机正态分布数组：",randomNormalArray)
    print()
    #36.7
    print("p36页第7题：利用NumPy函数sort()对数组元素排序，排序结果不覆盖原数组内容。")
    sortedArray = np.sort(randomNormalArray)
    print("排序后的数组：",sortedArray)
    print()
    #36.8
    print("p36页第8题：利用NumPy函数where()依次对数组元素进行逻辑判断。")
    result = np.where(sortedArray > 0, 'Yes', 'No')
    print(result)
    print()
    #36.9
    print("p36页第9题：利用NumPy的random.normal()函数生成2行5列的二维数组，数组元素服从均值为5、标准差为1的正态分布。")
    randomNormalArray1 = np.random.normal(5,1,5)
    randomNormalArray2 = np.random.normal(5,1,5)
    array2x5 = np.stack((randomNormalArray1,randomNormalArray2))
    print("2行5列数组，数组元素符合均值为5，标准差为1的正态分布：\n",array2x5)
    print()
    #36.10
    print("p36页第10题：利用eye()函数生成一个5行5列的单位矩阵Y。")
    matrix_Y = np.eye(5)
    print("5x5的单位矩阵Y：\n",matrix_Y)
    print()
    #36.11
    print("p36页第11题：利用dot()函数计算矩阵X和矩阵Y的矩阵乘积。")
    x = np.array([[1,2,3],[4,5,6]])
    print("矩阵X：\n",x)
    y = np.array([[2,3],[4,5],[6,7]])
    print("矩阵Y：\n",y)
    result = np.dot(x, y)
    print("结果为：\n",result)

#p37页习题
def p37():
    #37.12
    print("p37页第12题：导入NumPy的linalg模块，说明以下函数的功能inv()，svd()，eig()")
    import numpy as np
    from numpy import linalg
    print("inv()：计算矩阵的逆矩阵")
    print("svd()：奇异值分解")
    print("eig()：特征值和特征向量")
    print()
    #37.13
    print("p37页第13题：生成一个数组元素服从标准正态分布的二维数组X，\n       1) 生成mat, mat为X的转置矩阵与X的矩阵乘积；\n       2) 计算mat矩阵的逆；\n       3) 计算矩阵mat的特征值和特征向量；\n       4) 对矩阵mat做奇异值分解。")
    X = np.random.standard_normal((3, 3))
    print("X：\n",X)
    #37.13.1
    mat = np.dot(X.T, X)
    print("mat：\n",mat)
    #37.13.2
    inv_mat = np.linalg.inv(mat)
    print("mat的逆：\n",inv_mat)
    #37.13.3
    eig_val, eig_vec = np.linalg.eig(mat)
    print("mat的特征值：",eig_val)
    print("mat的特征向量：\n",eig_vec)
    #37.13.4
    U, S, V = np.linalg.svd(mat)
    print("U：\n",U)
    print("S：\n",S)
    print("V：\n",V)
    print()
    #37.14
    print("第37页第14题：\n    自行查找马氏距离的定义，并用NumPy实现计算一组向量[3,4]、[5,6]、[2,2]、[8,4]之间的马氏距离。")
    print("    马氏距离定义：马氏距离(Mahalanobis Distance)是度量学习中一种常用的距离指标，同欧氏距离、曼哈顿距离、汉明距离等一样被用作评定数据之间的相似度指标。但却可以应对高维线性分布的数据中各维度间非独立同分布的问题。")
    print("    马氏距离计算公式：D(x,y) = sqrt((x-y)T * S^(-1) * (x-y))")
    vector1 = np.array([3,4])
    vector2 = np.array([5,6])
    vector3 = np.array([2,2])
    vector4 = np.array([8,4])
    vectors = np.stack((vector1,vector2,vector3,vector4))
    print("     向量1,2之间距离：",get_mahalanobis(vectors,0,1))
    print("     向量1,3之间距离：",get_mahalanobis(vectors,0,2))
    print("     向量1,4之间距离：",get_mahalanobis(vectors,0,3))
    print("     向量2,3之间距离：",get_mahalanobis(vectors,1,2))
    print("     向量2,4之间距离：",get_mahalanobis(vectors,1,3))
    print("     向量3,4之间距离：",get_mahalanobis(vectors,2,3))

def get_mahalanobis(x, i, j):
    import numpy
    xT = x.T
    D = numpy.cov(xT)  # 求协方差矩阵
    invD = numpy.linalg.inv(D)  # 协方差逆矩阵
    assert 0 <= i < x.shape[0], "点 1 索引超出样本范围。"
    assert -1 <= j < x.shape[0], "点 2 索引超出样本范围。"
    x_A= x[i]
    x_B = x.mean(axis=0) if j == -1 else x[j]
    tp = x_A - x_B
    return numpy.sqrt(numpy.dot(numpy.dot(tp, invD), tp.T))

if __name__ == '__main__':
    p35()
    p36()
    p37()
