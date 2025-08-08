import numpy as np
from scipy.optimize import minimize
import time

def logistic_regression_loss(beta, X, y, r):
    #f(beta) = r||beta||^2 + ∑[ln(1+exp(x_i.dot(beta))) - y_i * (x_i.dot(beta))]
    z = X.dot(beta)
    loss = r * np.sum(beta ** 2) + np.sum(np.log1p(np.exp(z)) - y * z)
    return loss

def logistic_regression_grad(beta, X, y, r):
    #grad = 2r * beta + X.T · (σ(z) - y)
    z = X.dot(beta)
    sigma = 1.0 / (1.0 + np.exp(-z))
    grad = 2 * r * beta + X.T.dot(sigma - y)
    return grad

def data(p, n=20000, r=0.01):
    n_class = n // 2 
    p_dim = p  
    X1 = np.random.randn(n_class, p_dim)
    u = np.full(p_dim, 0.1)
    X2 = np.random.randn(n_class, p_dim) + u

    X1 = np.hstack([np.ones((n_class, 1)), X1])
    X2 = np.hstack([np.ones((n_class, 1)), X2])

    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_class), np.zeros(n_class)])

    beta0 = np.zeros(p_dim + 1)

    start_time = time.perf_counter()

    res = minimize(fun=logistic_regression_loss, x0=beta0, args=(X, y, r),
                   jac=logistic_regression_grad, method='BFGS', options={'disp': False})

    end_time = time.perf_counter()
    runtime = end_time - start_time

    res.runtime = runtime
    return res


for p in [11, 31, 101, 301]:
    print('p =',p)
    result = data(p)
    print("最优化结果信息:", result.message)
    print("迭代次数:", result.nit)
    print("最终目标函数值: {:.4f}".format(result.fun))
    print("最优参数 beta:", result.x)
    print("运行时间: {:.4f} 秒".format(result.runtime))
    print()
