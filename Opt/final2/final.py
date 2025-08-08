import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Lasso

def soft_thresh(x, k):
    return np.sign(x) * np.maximum(np.abs(x) - k, 0.0)

def admm_lasso(X, y, lam=0.05, sigma=1.0, rho=1.0, 
               max_iter=500, tol=1e-4):
    n, p = X.shape
    # precompute
    Xt = X.T
    XXt = X @ Xt
    # factorize (XXt + σ I)_n×n 便于 Woodbury
    L, lower = cho_factor(XXt + sigma * np.eye(n))
    Xty = Xt @ y

    # 初始化
    beta = np.zeros(p)
    gamma = np.zeros(p)
    mu = np.zeros(p)

    hist_obj = []
    for k in range(max_iter):
        # 1) β 更新 via Woodbury: β = (X^T X + σI)^(-1)(X^T y - μ + σ γ)
        v = Xty - mu + sigma * gamma
        # Woodbury: inv(X^T X + σI) v = (1/σ)(v - X^T ( (XX^T+σI)^-1 (X v) ))
        Xv = X @ v
        w = cho_solve((L, lower), Xv)
        beta_new = (v - Xt @ w) / sigma

        # 2) γ 更新：soft thresholding
        z = beta_new + mu / sigma
        gamma_new = soft_thresh(z, lam / sigma)

        # 3) μ 更新
        mu += sigma * (beta_new - gamma_new)

        # 记录收敛情况（目标值）
        obj = 0.5 * np.linalg.norm(X @ beta_new - y)**2 + lam * np.linalg.norm(beta_new, 1)
        hist_obj.append(obj)

        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta, gamma = beta_new, gamma_new

    return beta, hist_obj

if __name__ == "__main__":
    # 1. 生成数据
    np.random.seed(0)
    n, p = 1000, 10000
    X = np.random.randn(n, p)
    # 真正的 β̃
    beta_true = np.zeros(p)
    beta_true[:50] = np.random.randn(50)
    y = X @ beta_true + np.random.randn(n)

    # 2. ADMM 求解
    beta_admm, hist = admm_lasso(X, y, lam=0.05, sigma=1.0, max_iter=1000)

    # 3. sklearn Lasso 求解
    clf = Lasso(alpha=0.05, fit_intercept=False, max_iter=5000, tol=1e-4)
    clf.fit(X, y)
    beta_sk = clf.coef_

    # 4. 画图对比
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(hist)
    plt.title("ADMM Objective")
    plt.xlabel("iter")
    plt.ylabel("obj")

    plt.subplot(1,3,2)
    idx = np.arange(200)  # 只画前 200 个系数
    plt.plot(idx, beta_true[idx], label="true")
    plt.plot(idx, beta_admm[idx], '--', label="ADMM")
    plt.plot(idx, beta_sk[idx], ':', label="sklearn")
    plt.legend()
    plt.title("Coefficients Comparison")

    plt.subplot(1,3,3)
    plt.scatter(beta_sk, beta_admm, s=5, alpha=0.5)
    plt.plot([beta_sk.min(), beta_sk.max()],
             [beta_sk.min(), beta_sk.max()], 'r--')
    plt.xlabel("sklearn")
    plt.ylabel("ADMM")
    plt.title("scatter ADMM vs sklearn")

    plt.tight_layout()
    plt.show()

    # 5. 打印二者差异
    print("||β_ADMM - β_sklearn||_2 =", np.linalg.norm(beta_admm - beta_sk))