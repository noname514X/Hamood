import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
###########################################
np.random.seed(42)
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

    return np.where(a > tau, a - tau,           # if a > tau
                   np.where(a < -tau, a + tau,  # elif a < -tau
                           0.0))                # else

def ADMM(X, y, lambda_val, sigma, max_iter=5000, tol_abs=1e-6, tol_rel=1e-4, adaptive_rho=True):
    N, p = X.shape

    beta = np.zeros((p, 1))
    gamma = np.zeros((p, 1))
    mu = np.zeros((p, 1))
    
    # 预计算常量
    XT_y = X.T @ y
    XT_XT_woodbury_part = X @ X.T + sigma * np.identity(N)
    
    # 自适应参数调整
    rho = sigma
    tau_incr = 2.0
    tau_decr = 2.0
    mu_factor = 10.0

    history = {
        'beta': [],
        'primal_residual': [],
        'dual_residual': [],
        'obj_val': [],
        'rho': []
    }

    for k in range(max_iter):
        # Beta更新 - 使用当前rho值
        v_prime = XT_y - mu
        term_X_v_prime = X @ v_prime
        
        # 如果rho改变，需要重新计算Woodbury部分
        if k > 0 and adaptive_rho and abs(rho - history['rho'][-1]) > 1e-8:
            XT_XT_woodbury_part = X @ X.T + rho * np.identity(N)
            
        z_woodbury = np.linalg.solve(XT_XT_woodbury_part, term_X_v_prime)
        beta_new = (1.0 / rho) * (v_prime - X.T @ z_woodbury)

        # Gamma更新 - 使用优化的软阈值函数
        u = beta_new + (1.0 / rho) * mu
        tau = lambda_val / rho
        gamma_new = func(u, tau)

        # 对偶变量更新
        mu_new = mu + rho * (beta_new - gamma_new)

        # 计算残差
        primal_residual = np.linalg.norm(beta_new - gamma_new)
        dual_residual = np.linalg.norm(rho * (gamma_new - gamma))

        # 自适应调整rho
        if adaptive_rho and k > 0:
            if primal_residual > mu_factor * dual_residual:
                rho = tau_incr * rho
                mu_new = mu_new / tau_incr
            elif dual_residual > mu_factor * primal_residual:
                rho = rho / tau_decr
                mu_new = mu_new * tau_decr

        # 收敛条件
        eps_primal = np.sqrt(p) * tol_abs + tol_rel * np.maximum(np.linalg.norm(beta_new), np.linalg.norm(gamma_new))
        eps_dual = np.sqrt(p) * tol_abs + tol_rel * np.linalg.norm(mu_new)

        # 目标函数值
        obj_val = 0.5 * np.linalg.norm(X @ beta_new - y)**2 + lambda_val * np.linalg.norm(gamma_new, 1)
      
        # 记录历史
        history['beta'].append(beta_new.copy())
        history['obj_val'].append(obj_val)
        history['primal_residual'].append(primal_residual)
        history['dual_residual'].append(dual_residual)
        history['rho'].append(rho)

        if (k + 1) % 200 == 0 or k == 0:
            print(f"  迭代次数 {k+1}/{max_iter}: 原始残差 = {primal_residual:.6f}, 对偶残差 = {dual_residual:.6f}, 目标函数值 = {obj_val:.6f}, rho = {rho:.4f}")

        # 改进的收敛条件
        if primal_residual < eps_primal and dual_residual < eps_dual:
            print(f"ADMM 在 {k+1} 次迭代后收敛。")
            break
            
        beta = beta_new
        gamma = gamma_new
        mu = mu_new

    else:
        print(f"ADMM 达到最大迭代次数 {max_iter}")
    
    return beta_new, history

lambda_val_task = 0.05 
sigma_vals = [ 0.5, 1.0, 2.0, 5.0]
#sigma_vals = [10.0, 20.0, 30.0]
#sigma_vals = [100.0, 200.0, 300.0]
#sigma_vals = [1000.0, 2000.0, 3000.0]
max_iterations = 5000

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


alpha_sklearn = lambda_val_task / (2 * N) # Adjusted alpha

start_time_sklearn = time.time()
# Using tighter tolerance and more iterations for sklearn for a fairer comparison
lasso_sklearn = Lasso(alpha=alpha_sklearn, fit_intercept=False, max_iter=5000, tol=1e-6, random_state=42) 
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
    plt.plot(beta_admm_compare, 'b-', alpha=0.7, label='ADMM (σ=1.0) Beta') #
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

