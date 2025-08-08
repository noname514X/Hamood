import numpy as np
from scipy.optimize import minimize
import time

def logistic_regression_loss(beta, X, y, r):
    """
    目标函数: f(beta) = r||beta||^2 + ∑[ln(1+exp(x_i.dot(beta))) - y_i * (x_i.dot(beta))]
    """
    z = X.dot(beta)
    # 数值稳定性处理：使用 np.log1p(np.exp(z)) 替代直接计算 ln(1+exp(z))
    loss = r * np.sum(beta ** 2) + np.sum(np.log1p(np.exp(z)) - y * z)
    return loss

def logistic_regression_grad(beta, X, y, r):
    """
    梯度: grad = 2r * beta + X.T · (σ(z) - y)
    """
    z = X.dot(beta)
    sigma = 1.0 / (1.0 + np.exp(-z))
    grad = 2 * r * beta + X.T.dot(sigma - y)
    return grad

def run_experiment(p, n=20000, r=0.01, seed=0):
    """
    对于给定的 p 值实验：
    - n = 20000 样本，其中前 10,000 个为类别 1，后 10,000 为类别 0
    - 类别 1: x' ~ N(0, I)
    - 类别 0: x' ~ N(u, I)，u 为全 0.1 向量，维度为 p
    - 增广数据：在 x' 前添加常数项 1，即 x = [1, x']
    """
    np.random.seed(seed)
    n_class = n // 2  # 每个类别 10000
    p_dim = p       # x' 的维数

    # 类别 1: x' ~ N(0, I)
    X1 = np.random.randn(n_class, p_dim)
    # 类别 0: x' ~ N(u, I), u = (0.1, ..., 0.1)
    u = np.full(p_dim, 0.1)
    X2 = np.random.randn(n_class, p_dim) + u

    # 增广数据：添加截距 1
    X1 = np.hstack([np.ones((n_class, 1)), X1])
    X2 = np.hstack([np.ones((n_class, 1)), X2])

    # 合并数据与标签
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_class), np.zeros(n_class)])

    # 初始化 beta (维度为 p+1)
    beta0 = np.zeros(p_dim + 1)

    # 记录起始时间
    start_time = time.perf_counter()
    
    # 调用 scipy.optimize.minimize 使用 BFGS 方法求解
    res = minimize(fun=logistic_regression_loss, x0=beta0, args=(X, y, r),
                   jac=logistic_regression_grad, method='BFGS', options={'disp': False})
    
    # 记录结束时间，计算运行时间
    end_time = time.perf_counter()
    runtime = end_time - start_time
    
    # 将运行时间信息添加进返回结果中
    res.runtime = runtime
    return res

# 针对 p=11, 31, 101, 301 做实验，并打印每个实验的运行时间
for p in [11, 31, 101, 301]:
    print(f"----- 针对 p = {p} 的实验 -----")
    result = run_experiment(p)
    print("最优化结果信息:", result.message)
    print("迭代次数:", result.nit)
    print("最终目标函数值: {:.4f}".format(result.fun))
    print("最优参数 beta:", result.x)
    print("运行时间: {:.4f} 秒".format(result.runtime))
    print("-" * 50)
