import math
import numpy as np
import time
import matplotlib.pyplot as plt


def step666(min, max, f, grad, x, d, f0, dphi0, c1, c2, max_iteration=200):
    for i in range(max_iteration):
        candidate_step = (max + min) / 2.0
        x_new = x + candidate_step * d
        f_new = f(x_new)
        x_min = x + min * d
        f_min = f(x_min)

        if not np.isfinite(f_new):
            max = candidate_step
            continue

        if (f_new > f0 + c1 * candidate_step * dphi0) or (f_new >= f_min):
            max = candidate_step
        else:
            grad_new = grad(x_new)
            dphi = np.dot(grad_new, d)

            if not np.isfinite(dphi):
                max = candidate_step
                continue

            if dphi >= c2 * dphi0:
                return candidate_step
            if dphi * (max - min) >= 0:
                max = min
            min = candidate_step
    return candidate_step


def wolfe(f, grad, x, d, step0, step1, c1, c2, max_iteration=200):
    f0 = f(x)
    g0 = grad(x)
    dphi0 = np.dot(g0, d)

    step_previous = step0
    step = step1

    for i in range(max_iteration):
        x_new = x + step * d
        f_new = f(x_new)

        if not np.isfinite(f_new):
            step = step_previous / 2.0
            continue

        if (f_new > f0 + c1 * step * dphi0) or (i > 0 and f_new >= f(x + step_previous * d)):
            return step666(step_previous, step, f, grad, x, d, f0, dphi0, c1, c2)

        grad_new = grad(x_new)
        dphi = np.dot(grad_new, d)

        if not np.isfinite(dphi):
            step = step_previous / 2.0
            continue

        if dphi >= c2 * dphi0:
            return step

        step_previous = step
        step = min(step * 2.0, 100.0)  


    return step


def BFGS(f, grad, x0, max_iteration=200, tolerance=1e-5):
    starttime = time.perf_counter()

    n = len(x0)
    x = x0.copy()
    Hessian = np.eye(n)

    for i in range(max_iteration):
        g = grad(x)
        if not np.all(np.isfinite(g)):
            print(f"第 {i} 次迭代：梯度无效，停止")
            break

        if np.linalg.norm(g) < tolerance:
            print(f"第 {i} 次迭代：梯度范数 {np.linalg.norm(g)}，收敛")
            break

        Hessian_grad = np.dot(Hessian, g)
        direction = -Hessian_grad

        step = wolfe(f, grad, x, direction, 0.1, 0.1, 1e-4, 0.9)  # 减小初始步长

        x_new = x + step * direction
        if not np.all(np.isfinite(x_new)):
            break

        s = x_new - x
        grad_new = grad(x_new)
        y = grad_new - g

        ys = np.dot(y, s)
        if abs(ys) < 1e-10 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(s)):
            print(f"第 {i} 次迭代：跳过 Hessian 更新")
            x = x_new
            continue

        rho = 1.0 / ys
        if abs(rho) > 1e10:
            print(f"第 {i} 次迭代：rho 过大，跳过 Hessian 更新")
            x = x_new
            continue

        I = np.eye(n)
        syT = np.outer(s, y)
        ysT = np.outer(y, s)

        M1 = I - rho * syT
        M2 = I - rho * ysT
        ssT = np.outer(s, s)

        # 修正 Hessian 更新公式
        Hessian = np.dot(M1, np.dot(Hessian, M2)) + rho * ssT

        x = x_new

        print(f"第 {i} 次迭代：目标函数值 {f(x)}，梯度范数 {np.linalg.norm(grad(x))}")

    endtime = time.perf_counter()
    print(f"BFGS 运行时间：{endtime - starttime} 秒")
    print(f"最终目标函数值：{f(x)}")
    print(f"最终梯度范数：{np.linalg.norm(grad(x))}")
    print(f"最终参数：{x}")

    return x, f(x)


def data(n, p, u):
    n1 = n // 2
    n2 = n - n1
    p_prime = p - 1



    x_prime_part1 = np.random.randn(n1, p_prime)
    x_prime_part2 = np.random.randn(n2, p_prime) + u
    x_prime_full = np.vstack((x_prime_part1, x_prime_part2))


    x_prime_full = (x_prime_full - np.mean(x_prime_full, axis=0)) / (np.std(x_prime_full, axis=0) + 1e-10)

    intercept_col = np.ones((n, 1))
    X_final = np.hstack((intercept_col, x_prime_full))

    y_labels_part1 = np.ones(n1, dtype=int)
    y_labels_part2 = np.zeros(n2, dtype=int)
    y_final = np.concatenate((y_labels_part1, y_labels_part2))

    return X_final, y_final


def f(beta):
    reg = r * np.dot(beta, beta)
    loss = 0
    for i in range(n):
        v = np.dot(X[i], beta)
        v = np.clip(v, -500, 500)
        try:
            loss += math.log(1 + math.exp(v)) - y[i] * v
        except OverflowError:
            loss += v - y[i] * v
    if not np.isfinite(loss):
        return np.inf
    return reg + loss


def grad_f(beta):
    grad = 2 * r * beta
    for i in range(n):
        v = np.dot(X[i], beta)
        v = np.clip(v, -500, 500)
        try:
            p = math.exp(v) / (1 + math.exp(v))
        except OverflowError:
            p = 1.0 if v > 0 else 0.0
        diff = p - y[i]
        grad += diff * X[i]
    if not np.all(np.isfinite(grad)):
        return np.zeros_like(grad) 
    return grad


if __name__ == "__main__":
    global X, y, r, n
    n = 20000
    r = 0.01
    p = [101, 301]

    for p1 in p:
        print(f"p = {p1}")
        p_ = p1 - 1
        u = np.full(p_, 0.1)

        X, y = data(n=n, p=p1, u=u)

        beta = np.random.randn(p1) * 0.01 

        time_start = time.perf_counter()
        result = BFGS(f, grad_f, beta, max_iteration=200)
        print(f"{result}")
        time_end = time.perf_counter()
        print(f"运行时间: {time_end - time_start:.2f} 秒")
        print()