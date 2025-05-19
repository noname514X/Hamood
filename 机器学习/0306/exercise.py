import numpy as np

#p15页练习
def p15():
    #15.2 将整数列表[1,2,3]转换为Numpy数组a,并查看数组对象a及a的类型。
    a = np.array([1,2,3])
    print(a)
    print(type(a))
    print()
    #15.3 
    b = np.array([1.2,2.3,3.4])
    print(b)
    print(type(b))
    print()
    #15.4 生成2*3的全0数组
    c = np.zeros((2,3))
    print(c)
    print()
    #15.5 生成2*3的全1数组
    d = np.ones((2,3))
    print(d)
    print()
    #15.6 生成2*3未初始化的随机数数组
    e = np.empty((2,3))
    print(e)
    print()
    #15.7 
    f = np.arange(10,30,5)
    print(f)
    print()
    #15.8 生成3*2的符合(0,1)均匀分布的随机数数组
    g = np.random.rand(3,2)
    print(g)
    print()
    #15.9 生成0到2范围内长度为5的整数数组
    h = np.linspace(0,2,5)
    print(h)
    print()
    #15.10 生成长度为3的符合标准正态分布的随机数数组
    i = np.random.randn(3)
    print(i)
    print()

#p16,p17练习
def p16():
    #16.11 利用np.arange创建元素分别为[0,1,4,9,16,25,36,49,64,81]的数组a
    a = np.arange(10)**2
    print(a)
    print()
    #16.12 获取数组a的第三个元素
    print(a[2])
    print()
    #16.13 获取数组a的第二到第四个元素
    print(a[1:4])
    print() 
    #16.14 翻转一维数组a
    print(a[::-1])
    print()
    #16.15 创建一个3*3的二维数组b，符合均匀分布的随机数
    b = np.random.uniform(0,1,(3,3))
    print(b)
    print()
    #16.16 获取数组b的第二行第三列
    print(b[1,2])
    print()
    #16.17 获取数组b的第二列
    print(b[:,1])
    print()
    #16.18 获取数组b的第三列前两行
    print(b[:2,2])
    print()
    #16.19 创建一个3*4的符合(0,1)均匀分布的随机数组a,逐元素乘10后向下去整。
    c = np.floor(np.random.rand(3,4)*10)
    print(c)
    print()
    #17.20 将数组a展平
    print(c.ravel())
    print()
    #17.21 将数组a变换为2*6数组
    print(c.reshape(2,6))
    print()







if __name__ == '__main__': 
    p16()