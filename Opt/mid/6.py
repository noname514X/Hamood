import math
import numpy as np
import time
import matplotlib.pyplot as plt


def step666(min, max, f, grad, x,d, f0, dphi0, c1, c2, max_iteration = 200):
    for i in range(max_iteration):
        candidate_step = (max + min) / 2.0
        x_new = x + candidate_step * d
        f_new = f(x_new)
        x_min = x + min * d
        f_min = f(x_min)

        if(f_new > f0 + c1 * candidate_step * dphi0) or (f_new >= f_min):
            max = candidate_step
        else:
            grad_new = grad(x_new)
            dphi = np.dot(grad_new, d)

            if(dphi >= c2 * dphi0):
                return candidate_step
            if(dphi * (max - min) >= 0):
                max = min
            min = candidate_step
    return candidate_step

def wolfe(f, grad, x, d, step0, step1, c1, c2, max_iteration = 200):
    f0 = f(x)
    g0 = grad(x)
    dphi0 = np.dot(g0, d)

    step_previou = step0
    step = step1
    step_max = 1e3

    for i in range(max_iteration):
        x_new = x + step * d
        f_new = f(x_new)

        if(f_new > f0 + c1 * step * dphi0) or i > 0 and f_new >= f(x + step_previous * d):
            return step666(step_previou, step, f, grad, x, d, f0, dphi0, c1, c2)

        grad_new = grad(x_new)
        dphi = np.dot(grad_new, d)

        if(dphi >= c2 * dphi0):
            return step
        
        step_previous = step
        step = step * 2.0
        
    return step

def BFGS(f, grad, x0 ,max_iteration = 200, tolerance= 1e-5):
    starttime = time.perf_counter()

    n = len(x0)
    x = x0[:]
    Hessian = np.eye(n)

    for i in range(max_iteration):
        g = grad(x)
        if np.linalg.norm(g) < tolerance:
            print("迭代", i, "次后满足收敛条件，梯度范数:", np.linalg.norm(g))
            break
    
        Hessian_grad = np.dot(Hessian, g)
        direction = -Hessian_grad

        step = wolfe(f, grad, x, direction, 1.0, 2.0, 1e-4, 0.9)

        x_new = x + step * direction
        s = x_new - x

        grad_new = grad(x_new)
        y = grad_new - g

        ys = np.dot(y, s)
        if abs(ys) < 1e-10:
            print("y和s的内积为0，无法更新Hessian矩阵")
            return x_new
        rho = 1.0 / ys

        I = np.eye(n)
        syT = np.outer(s, y)
        ysT = np.outer(y, s)

        M1 = I - rho * syT
        M2 = I - rho * ysT

        temp = np.dot(M1, np.dot(Hessian, M2))

        ssT = np.outer(s, s)

        term1 = temp    
        term2 = rho * temp
        Hessian = term1 + term2

        x = x_new

        print("迭代", i, "次，当前目标函数值:", f(x), "当前梯度范数:", np.linalg.norm(grad(x)))
    endtime = time.perf_counter()
    print("BFGS算法运行时间:", endtime - starttime, "秒")
    print("最终目标函数值:", f(x))
    print("最终梯度范数:", np.linalg.norm(grad(x)))
    print("最终参数:", x)

    return x, f(x)

def data(n, p, u):
    n1 = n // 2 
    n2 = n - n1  
    p_prime = p - 1 

    if len(u) != p_prime:
        raise ValueError(f"均值向量u的长度({len(u)})必须等于p-1({p_prime})")

    x_prime_part1 = np.random.randn(n1, p_prime)  
    x_prime_part2 = np.random.randn(n2, p_prime) + u 
    x_prime_full = np.vstack((x_prime_part1, x_prime_part2))

    intercept_col = np.ones((n, 1))
    X_final = np.hstack((intercept_col, x_prime_full))

    y_labels_part1 = np.ones(n1, dtype=int)  
    y_labels_part2 = np.zeros(n2, dtype=int) 
    y_final = np.concatenate((y_labels_part1, y_labels_part2))
    
    return X_final, y_final

# def f(beta):
# #f(beta) = r * ||beta||^2 + sum_i [ ln(1 + exp(x_i^T beta)) - y_i*(x_i^T beta) ]
#     reg = r * np.dot(beta, beta)
#     loss = 0
#     for i in range(n):
#         v = np.dot(X[i], beta)
#         if v > 700:
#             loss += v - y[i] * v
#         else:
#             try:
#                 loss += math.log(1 + math.exp(v)) - y[i] * v
#             except OverflowError:
#                 loss += v - y[i] * v
#     return reg + loss

# def grad_f(beta):
# #grad_f(beta) = 2r * beta + sum_i [(exp(v)/(1+exp(v)) - y_i) * x_i]
#     grad = 2*r*beta
#     for i in range(n):
#         v = np.dot(X[i], beta)
#         if v > 700:
#             p = 1.0
#         else:
#             try:
#                 p = math.exp(v) / (1 + math.exp(v))
#             except OverflowError:
#                 p = 1.0
#         diff = p - y[i]
#         grad += diff * X[i]
#     return grad

# def sigmoid(x):
#     if x > 0:
#         return 1 / (1 + np.exp(-x))
#     else:
#         return np.exp(x) / (1 + np.exp(x))
    
def f(beta):
    reg = r * np.dot(beta, beta)
    v = np.dot(X, beta)
    loss = np.sum(np.log1p(np.exp(v)) - y * v)
    return reg + loss

def grad_f(beta):
    grad = 2 * r * beta
    v = np.dot(X, beta)
    p = 1 / (1 + np.exp(-v)) 
    diff = p - y
    grad += np.dot(diff, X)
    return grad

if __name__ == "__main__":
    global X,y,r,n
    n = 20000
    r = 0.01
    p = [11, 31, 101, 301]

    for p1 in p:
            print("p =", p1)
            p_ = p1 - 1
            u = np.full(p_, 0.1)

            X, y = data(n = n, p = p1, u = u)

            beta = np.zeros(p1)

            print("开始运行p =", p1, "时的BFGS优化")

            BFGS(f, grad_f, beta, max_iteration = 200)

            print()
