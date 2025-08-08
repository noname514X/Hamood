import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
###########################################
np.random.seed(23089056)
N = 1000
p = 10000
X = np.random.randn(N, p)
beta_true = np.zeros((p, 1))
v = np.random.randn(50, 1)
beta_true[:50] = v**2 
epsilon = np.random.randn(N, 1)
y = X @ beta_true + epsilon

##########################################


#λ,ß,γ,μ

# a = 0.5 * np.sum((np.dot(X, beta) - y)**2)
# b = lamda * np.sum(np.abs(gamma))
def func(a, tau):
    if tau < 0:
        raise ValueError("tau必须是非负数")

    if np.isscalar(a): 
        if a > tau:
            return a - tau
        elif a < -tau:
            return a + tau
        else: # -tau <= a <= tau
            return 0.0
    else: 
        result = np.zeros_like(a, dtype=float) 
        for i in range(a.shape[0]):
            if a[i] > tau:
                result[i] = a[i] - tau
            elif a[i] < -tau:
                result[i] = a[i] + tau
            else: # -tau <= a[i] <= tau
                result[i] = 0.0
        return result

def ADMM(X, y, lambda_val, sigma, max_iter=2000, tol_abs=1e-4, tol_rel=1e-4):
    N, p = X.shape

    beta = np.zeros((p, 1))  #初始化为零向量
    gamma = np.zeros((p, 1)) #初始化为零向量
    mu = np.zeros((p, 1)) 
    #(X.T @ X + sigma * I)^(-1) = sigma^(-1) * (I - X.T * (X @ X.T + sigma * I)^(-1) * X)
    #(X.T @ X + sigma * I)^(-1) * v = sigma^(-1) * (v - X.T * (X @ X.T + sigma * I)^(-1) * (X @ v))
    XT_y = X.T @ y #X.T @ y
    XT_XT_woodbury_part = X @ X.T + sigma * np.identity(N) #X @ X.T + sigma * I

    history = {
        'beta': [],
        'primal_residual': [],
        'dual_residual': [],
        'obj_val': []
    }

    for k in range(max_iter):
        #ß_new = (1.0 / sigma) * (XT_y - X.T @ mu)
        v_prime = XT_y - mu + sigma * gamma # X.T @ y - mu + sigma * gamma
        term_X_v_prime = X @ v_prime
        z_woodbury = np.linalg.solve(XT_XT_woodbury_part, term_X_v_prime)
        beta_new = (1.0 / sigma) * (v_prime - X.T @ z_woodbury)

        #gamma_new = func(ßnew + (1.0 / sigma) * μ, λ / σ)
        u = beta_new + (1.0 / sigma) * mu
        tau = lambda_val / sigma
        gamma_new = func(u, tau)

        #μ_new = μ + σ * (ß_new - γ_new)
        mu_new = mu + sigma * (beta_new - gamma_new)

        primal_residual = np.linalg.norm(beta_new - gamma_new)
        dual_residual = np.linalg.norm(sigma * (gamma_new - gamma))

        eps_primal = np.sqrt(p) * tol_abs + tol_rel * np.maximum(np.linalg.norm(beta_new), np.linalg.norm(gamma_new))
        eps_dual = np.sqrt(p) * tol_abs + tol_rel * np.linalg.norm(mu_new)

        #0.5 * ||X * beta_new - y||^2 + λ * ||γ_new||_1
        obj_val = 0.5 * np.linalg.norm(X @ beta_new - y)**2 + lambda_val * np.linalg.norm(gamma_new, 1)
      
        history['beta'].append(beta_new.copy())
        history['obj_val'].append(obj_val)
        history['primal_residual'].append(primal_residual)
        history['dual_residual'].append(dual_residual)

        if (k + 1) % 100 == 0 or k == 0:
            print(f"  迭代次数 {k+1}/{max_iter}: 原始残差 = {primal_residual:.6f}, 对偶残差 = {dual_residual:.6f}, 目标函数值 = {obj_val:.6f}")

        if primal_residual < eps_primal and dual_residual < eps_dual:
            print(f"ADMM 在 {k+1} 次迭代后收敛。")
            break
            
        beta = beta_new
        gamma = gamma_new
        mu = mu_new

    else:
        print(f"ADMM 达到最大迭代次数 {max_iter}")
    
    return beta_new,history

lambda_val_task = 0.05 
sigma_vals = [0.05, 0.1, 0.15, 0.5, 1.0, 1.5, 3.0, 5.0, 10.0]
#sigma_vals = [10.0, 20.0, 30.0]
#sigma_vals = [100.0, 200.0, 300.0]
#sigma_vals = [1000.0, 2000.0, 3000.0]
max_iterations = 10000

admm_results_corrected = {}

for sigma in sigma_vals:
    print(f"\n--- 运行 ADMM (sigma = {sigma}) ---")
    start_time = time.time()
    beta_admm, history = ADMM(X, y, lambda_val_task, sigma, max_iter=max_iterations)
    end_time = time.time()
    
    admm_results_corrected[sigma] = {
        'beta': beta_admm,
        'history': history,
        'time': end_time - start_time
    }
    print(f"ADMM (sigma={sigma}) 计算耗时: {end_time - start_time:.4f} 秒")
    print(f"ADMM (sigma={sigma}) 求解的 Beta 中非零系数数量: {np.sum(np.abs(beta_admm) > 1e-6)}")
    print(f"ADMM (sigma={sigma}) 求解的 Beta :")
    print(beta_admm[beta_admm != 0].flatten()[:10])  

#######
#######
#######



print("\n--- 与 sklearn.linear_model.Lasso 比较(sigma = 1) ---")


alpha_sklearn = lambda_val_task

start_time_sklearn = time.time()
lasso_sklearn = Lasso(alpha=alpha_sklearn, fit_intercept=False, max_iter=10000, tol=1e-6, random_state=42) 
lasso_sklearn.fit(X, y.ravel()) 
end_time_sklearn = time.time()

beta_sklearn = lasso_sklearn.coef_.reshape(p, 1)

print(f"Sklearn Lasso 计算耗时: {end_time_sklearn - start_time_sklearn:.4f} 秒")
print(f"Sklearn Lasso 求解的 Beta 中非零系数数量: {np.sum(np.abs(beta_sklearn) > 1e-6)}")
print(f"Sklearn Lasso 求解的 Beta :")
print(beta_sklearn[beta_sklearn != 0].flatten()[:10]) 

plt.plot(beta_true, 'k.', markersize=5, label='True Beta')
beta_admm_compare = admm_results_corrected[1.0]['beta']
if 'beta_admm_compare' in locals(): #
    plt.plot(beta_admm_compare, 'b-', alpha=0.7, label='ADMM (σ=1.0 ) Beta') #
plt.plot(beta_sklearn, 'r--', alpha=0.7, label='Sklearn Lasso Beta') #
plt.title('Comparison of Beta Coefficients')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.legend()
plt.grid(True)
plt.show()





print("\n--- 不同sigma下的ADMM(图像) ---")
plt.figure(figsize=(10, 6))


sigma_values = []
computation_times = []

for sigma in sigma_vals:
    sigma_values.append(sigma)
    computation_times.append(admm_results_corrected[sigma]['time'])

# 柱状图 比较计算时间
plt.subplot(1, 2, 1)
bars = plt.bar(range(len(sigma_values)), computation_times, alpha=0.7, color='skyblue', edgecolor='navy')
plt.xlabel('Sigma Values')
plt.ylabel('Computation Time (seconds)')
plt.title('ADMM Computation Time vs Sigma')
plt.xticks(range(len(sigma_values)), [f'{s}' for s in sigma_values], rotation=45)
plt.grid(True, alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(computation_times),
             f'{height:.3f}s', ha='center', va='bottom', fontsize=9)

# 绘制线图展示趋势
plt.subplot(1, 2, 2)
plt.plot(sigma_values, computation_times, 'o-', linewidth=2, markersize=8, color='red')
plt.xlabel('Sigma Values(log scale)')
plt.ylabel('Computation Time (seconds)')
plt.title('ADMM Computation Time Trend')
plt.grid(True, alpha=0.3)
plt.xscale('log')  # 对数尺度


for i, (x, y) in enumerate(zip(sigma_values, computation_times)):
    plt.annotate(f'{y:.3f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))

# 绘制目标函数值收敛曲线 (展示不同 sigma 时的收敛速度)
for sigma in sigma_vals:
    if 'history' in admm_results_corrected[sigma] and 'obj_val' in admm_results_corrected[sigma]['history']:
        plt.plot(admm_results_corrected[sigma]['history']['obj_val'], label=f'ADMM (σ={sigma})')
plt.title('Objective Function Value Convergence(ADMM)')
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

