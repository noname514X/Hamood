# 导入必要的库，numpy 用于数学计算
import numpy as np
# 导入 math 库，有时初学者会混用 math 和 numpy 的函数
import math 
# 增加时间库，用于观察不同维度下的运行时间 (可选，但对比较有帮助)
import time

# --- 定义一些可能会用到的帮助函数 ---

# Sigmoid 函数 (保持不变)
def calculate_sigmoid(z):
    try:
        # 使用 np.exp 可能更稳定且高效，但 math.exp 更符合某些初学者习惯
        # result = 1.0 / (1.0 + np.exp(-z)) 
        result = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        result = 1e-15 
    if result < 1e-15:
        result = 1e-15
    if result > 1 - 1e-15:
        result = 1 - 1e-15
    return result

# 计算目标函数 J(beta) 的值 (保持不变, 使用循环)
def calculate_total_objective(current_beta, feature_matrix, labels, reg_param_r):
    num_samples = feature_matrix.shape[0] 
    l2_term = reg_param_r * np.sum(current_beta * current_beta) 
    log_likelihood = 0.0
    for i in range(num_samples):
        x_i = feature_matrix[i, :] 
        y_i = labels[i]          
        linear_combination = np.dot(x_i, current_beta)
        predicted_prob = calculate_sigmoid(linear_combination)
        epsilon = 1e-7 
        term1 = -y_i * math.log(predicted_prob + epsilon)
        term2 = -(1 - y_i) * math.log(1 - predicted_prob + epsilon)
        log_likelihood += (term1 + term2)
    total_objective_value = l2_term + log_likelihood
    return total_objective_value

# 计算目标函数 J(beta) 的梯度 (保持不变, 使用循环)
def calculate_gradient_vector(current_beta, feature_matrix, labels, reg_param_r):
    num_samples = feature_matrix.shape[0]
    num_features = feature_matrix.shape[1]
    l2_gradient = 2 * reg_param_r * current_beta
    likelihood_gradient = np.zeros(num_features) 
    for i in range(num_samples):
        x_i = feature_matrix[i, :] 
        y_i = labels[i]          
        linear_combination = np.dot(x_i, current_beta)
        predicted_prob = calculate_sigmoid(linear_combination)
        error = predicted_prob - y_i
        likelihood_gradient += error * x_i
    total_gradient = l2_gradient + likelihood_gradient
    return total_gradient

# --- 实现线搜索 (Line Search) --- (保持不变)
def wolfe_line_search(beta, direction, X, y, r, grad_current, 
                      objective_func, gradient_func, 
                      c1=1e-4, c2=0.9, alpha_init=1.0, max_ls_iters=20):
    alpha = alpha_init 
    f_k = objective_func(beta, X, y, r)
    grad_dot_direction = np.dot(grad_current, direction)
    for i in range(max_ls_iters):
        beta_new = beta + alpha * direction
        f_new = objective_func(beta_new, X, y, r)
        if f_new > f_k + c1 * alpha * grad_dot_direction:
            alpha = alpha * 0.5 
            continue 
        grad_new = gradient_func(beta_new, X, y, r)
        grad_new_dot_direction = np.dot(grad_new, direction)
        if grad_new_dot_direction < c2 * grad_dot_direction:
            alpha = alpha * 0.5 
            continue 
        return alpha
    print("Warning: Line search could not find a suitable step size satisfying Wolfe conditions.")
    return alpha 

# --- BFGS 算法实现 --- (保持不变, 可能需要调整迭代次数和容忍度)
# 注意：对于大数据集和高维度，原循环计算目标函数和梯度会非常慢！
# 这是符合“可以运行缓慢”的要求，但可能需要很长时间。
def my_bfgs_optimizer(feature_matrix, labels, reg_param_r, 
                       initial_beta_guess, max_iterations=100, convergence_tolerance=1e-5): 
    
    num_features = feature_matrix.shape[1]
    beta_k = np.copy(initial_beta_guess) 
    H_k = np.identity(num_features) 
    gradient_k = calculate_gradient_vector(beta_k, feature_matrix, labels, reg_param_r)
    
    print("--- Starting BFGS Optimization ---")
    initial_obj = calculate_total_objective(beta_k, feature_matrix, labels, reg_param_r)
    initial_grad_norm = np.linalg.norm(gradient_k)
    print(f"Initial Objective Value: {initial_obj:.6f}")
    print(f"Initial Gradient Norm: {initial_grad_norm:.6f}")
    
    iteration_count = 0
    start_time = time.time() # 开始计时

    while iteration_count < max_iterations:
        gradient_norm = np.linalg.norm(gradient_k)
        if gradient_norm < convergence_tolerance:
            print(f"\nConvergence criterion met at iteration {iteration_count}.")
            break
            
        search_direction_pk = -np.dot(H_k, gradient_k)
        
        step_size_alphak = wolfe_line_search(
            beta=beta_k, direction=search_direction_pk, X=feature_matrix, y=labels, r=reg_param_r, 
            grad_current=gradient_k, objective_func=calculate_total_objective, gradient_func=calculate_gradient_vector
        )
        
        if step_size_alphak < 1e-10:
            print("Warning: Step size from line search is too small. Stopping optimization.")
            break
            
        beta_k_plus_1 = beta_k + step_size_alphak * search_direction_pk
        gradient_k_plus_1 = calculate_gradient_vector(beta_k_plus_1, feature_matrix, labels, reg_param_r)
        s_k = beta_k_plus_1 - beta_k
        y_k = gradient_k_plus_1 - gradient_k
        y_k_transpose_dot_s_k = np.dot(y_k.T, s_k)
        
        if y_k_transpose_dot_s_k > 1e-8: 
            rho_k = 1.0 / y_k_transpose_dot_s_k
            identity_matrix = np.identity(num_features)
            term1 = identity_matrix - rho_k * np.outer(s_k, y_k) 
            term2 = identity_matrix - rho_k * np.outer(y_k, s_k)
            term3 = rho_k * np.outer(s_k, s_k)
            H_k_plus_1 = np.dot(term1, np.dot(H_k, term2)) + term3
            H_k = H_k_plus_1
        else:
            print(f"Warning: Skipping BFGS update at iteration {iteration_count} because y_k^T * s_k = {y_k_transpose_dot_s_k:.2e} is not positive.")

        beta_k = beta_k_plus_1
        gradient_k = gradient_k_plus_1
        iteration_count += 1
        
        # 减少打印频率，因为计算目标函数本身就很慢
        if iteration_count % 20 == 0 or iteration_count == 1: 
            current_time = time.time()
            elapsed_time = current_time - start_time
            # 计算目标值和梯度范数会显著增加时间，仅在必要时计算
            # obj_val = calculate_total_objective(beta_k, feature_matrix, labels, reg_param_r)
            # grad_norm_val = np.linalg.norm(gradient_k)
            # print(f"Iteration {iteration_count}: Grad Norm={grad_norm_val:.6f}, Step Size={step_size_alphak:.4f}, Time={elapsed_time:.2f}s")
            # 只打印梯度范数和步长，避免重复计算目标函数值
            grad_norm_val = np.linalg.norm(gradient_k) # 这个在循环开始已经计算了
            print(f"Iteration {iteration_count}: Current Grad Norm={gradient_norm:.6f}, Step Size={step_size_alphak:.4f}, Time={elapsed_time:.2f}s")


    end_time = time.time() # 结束计时
    total_time = end_time - start_time

    if iteration_count == max_iterations:
        print(f"\nMaximum number of iterations ({max_iterations}) reached.")
        
    print("--- Optimization Finished ---")
    final_objective = calculate_total_objective(beta_k, feature_matrix, labels, reg_param_r)
    final_gradient_norm = np.linalg.norm(gradient_k)
    print(f"Final Beta (coefficients): {beta_k}")
    print(f"Final Objective Value: {final_objective:.6f}")
    print(f"Final Gradient Norm: {final_gradient_norm:.6f}")
    print(f"Total Optimization Time: {total_time:.2f} seconds")
    
    return beta_k

# --- 新增：根据作业要求生成数据的函数 ---
def generate_specific_data(n_samples, p_dimension, mean_vector_u):
    """
    根据作业要求生成特定的数据集。
    n_samples: 总样本数 (n)
    p_dimension: 总特征维度 (p, 包含截距)
    mean_vector_u: 第二部分数据特征的均值向量 (长度 p-1)
    """
    print(f"Generating data: n={n_samples}, p={p_dimension}...")
    
    n1 = n_samples // 2 # 前半部分样本数
    n2 = n_samples - n1 # 后半部分样本数
    p_prime = p_dimension - 1 # 实际特征维度 (不含截距)
    
    if len(mean_vector_u) != p_prime:
        raise ValueError(f"Length of mean_vector_u ({len(mean_vector_u)}) must be equal to p-1 ({p_prime})")

    # 生成特征 x'_i
    # 使用 numpy.random.randn 生成标准正态分布 N(0, I)
    # np.random.seed(42) # 如果需要可复现的结果，可以设置种子

    # 前半部分数据: N(0, I)
    x_prime_part1 = np.random.randn(n1, p_prime)
    
    # 后半部分数据: N(u, I)
    # 通过 N(0, I) + u 实现
    x_prime_part2 = np.random.randn(n2, p_prime) + mean_vector_u
    
    # 合并 x' 部分
    x_prime_full = np.vstack((x_prime_part1, x_prime_part2))
    
    # 创建截距列
    intercept_col = np.ones((n_samples, 1))
    
    # 最终特征矩阵 X = [1, x']
    X_final = np.hstack((intercept_col, x_prime_full))
    
    # 生成标签 y
    y_labels_part1 = np.ones(n1, dtype=int) # 前半部分 y = 1
    y_labels_part2 = np.zeros(n2, dtype=int) # 后半部分 y = 0
    y_final = np.concatenate((y_labels_part1, y_labels_part2))
    
    print("Data generation complete.")
    return X_final, y_final


# --- 主程序入口 ---
if __name__ == "__main__":
    
    print("Running Logistic Regression with L2 Regularization using custom BFGS.")
    
    # --- 设置全局参数 ---
    n_total_samples = 20000
    regularization_r_value = 0.01
    # 可能需要增加迭代次数和调整容忍度，因为数据量变大
    bfgs_max_iterations = 100 # 保持100次，或者适当增加如 200?
    bfgs_tolerance = 1e-4 # 稍微放宽容忍度，加速收敛或避免过早停止

    # --- 定义需要测试的维度 p ---
    p_dimensions_to_test = [11, 31, 101, 301]

    # --- 循环运行不同维度 ---
    for p_dim in p_dimensions_to_test:
        print(f"\n===== Running for p = {p_dim} =====")
        
        # 计算实际特征维度 p' = p - 1
        p_prime_dim = p_dim - 1
        
        # 定义第二部分数据的均值向量 u = (0.1, ..., 0.1)
        # 使用 np.full 创建一个长度为 p_prime_dim，值都是 0.1 的数组
        mean_u_vector = np.full(p_prime_dim, 0.1) 
        
        # --- 生成数据 ---
        X_generated, y_generated = generate_specific_data(
            n_samples=n_total_samples, 
            p_dimension=p_dim, 
            mean_vector_u=mean_u_vector
        )
        
        # --- 设置初始 beta ---
        initial_beta_guess = np.zeros(p_dim) 
        
        # --- 调用优化器 ---
        print(f"Starting optimization for p = {p_dim}...")
        optimized_beta = my_bfgs_optimizer(
            feature_matrix=X_generated, 
            labels=y_generated, 
            reg_param_r=regularization_r_value, 
            initial_beta_guess=initial_beta_guess,
            max_iterations=bfgs_max_iterations, 
            convergence_tolerance=bfgs_tolerance 
        )
        print(f"===== Finished running for p = {p_dim} =====")

    print("\nAll specified dimensions have been processed.")