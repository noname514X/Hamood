import numpy as np
import math
import time
import matplotlib.pyplot as plt


def sigmoid(z):
    try:
        result = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        result = 1e-15
    if result < 1e-15:
        result = 1e-15  
    if result > 1 - 1e-15:
        result = 1 - 1e-15
    return result

def objective(current_beta, feature_matrix, labels, reg_param_r):
    num_samples = feature_matrix.shape[0] 
    l2_term = reg_param_r * np.sum(current_beta * current_beta) 
    log_likelihood = 0.0
    for i in range(num_samples):
        x_i = feature_matrix[i, :] 
        y_i = labels[i]          
        linear_combination = np.dot(x_i, current_beta)
        predicted_prob = sigmoid(linear_combination)
        epsilon = 1e-7 
        term1 = -y_i * math.log(predicted_prob + epsilon)
        term2 = -(1 - y_i) * math.log(1 - predicted_prob + epsilon)
        log_likelihood += (term1 + term2)
    total_objective_value = l2_term + log_likelihood
    return total_objective_value

def gradient_vector(current_beta, feature_matrix, labels, reg_param_r):
    num_samples = feature_matrix.shape[0]
    num_features = feature_matrix.shape[1]
    l2_gradient = 2 * reg_param_r * current_beta
    likelihood_gradient = np.zeros(num_features) 
    for i in range(num_samples):
        x_i = feature_matrix[i, :] 
        y_i = labels[i]          
        linear_combination = np.dot(x_i, current_beta)
        predicted_prob = sigmoid(linear_combination)
        error = predicted_prob - y_i
        likelihood_gradient += error * x_i
    total_gradient = l2_gradient + likelihood_gradient
    return total_gradient

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
    print("线搜索无法找到满足 Wolfe 条件的合适步长")
    return alpha 

def BFGS(feature_matrix, labels, reg_param_r, initial_beta_guess, max_iterations=100, convergence_tolerance=1e-5): 
    
    num_features = feature_matrix.shape[1]
    beta_k = np.copy(initial_beta_guess) 
    H_k = np.identity(num_features) 
    gradient_k = gradient_vector(beta_k, feature_matrix, labels, reg_param_r)
    
    #print("BFGS 启动！")
    initial_obj = objective(beta_k, feature_matrix, labels, reg_param_r)
    initial_grad_norm = np.linalg.norm(gradient_k)
    print("初始目标函数值 {:.6f}".format(initial_obj))
    print("初始梯度范数: {:.6f}".format(initial_grad_norm))
    

    iteration_count = 0
    start_time = time.time() 

    while iteration_count < max_iterations:
        gradient_norm = np.linalg.norm(gradient_k)
        if gradient_norm < convergence_tolerance:
            print("在第", iteration_count, "次迭代时满足收敛条件。")
            break
            
        search_direction_pk = -np.dot(H_k, gradient_k)
        
        step_size_alphak = wolfe_line_search(
            beta=beta_k, direction=search_direction_pk, X=feature_matrix, y=labels, r=reg_param_r, 
            grad_current=gradient_k, objective_func=objective, gradient_func=gradient_vector
        )
        
        if step_size_alphak < 1e-10:
            print("线搜索的步长过小。停止。")
            break
            
        beta_k_plus_1 = beta_k + step_size_alphak * search_direction_pk
        gradient_k_plus_1 = gradient_vector(beta_k_plus_1, feature_matrix, labels, reg_param_r)
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
            print("跳过第{}次迭代Hessian更新，因为 y_k^T * s_k = {:.2e} 不是正数。".format(iteration_count, y_k_transpose_dot_s_k))

        beta_k = beta_k_plus_1
        gradient_k = gradient_k_plus_1
        iteration_count += 1
        

        if iteration_count % 20 == 0 or iteration_count == 1: 
            current_time = time.time()
            elapsed_time = current_time - start_time
            grad_norm_val = np.linalg.norm(gradient_k) 
            print(f"第{iteration_count}次优化，梯度范数：{gradient_norm:.6f}, 步长：{step_size_alphak:.4f}, 耗时：{elapsed_time:.2f}s")


    end_time = time.time() 
    total_time = end_time - start_time

    if iteration_count == max_iterations:
        print("已达到最大迭代次数", max_iterations)
        
    print("运行结束")
    final_objective = objective(beta_k, feature_matrix, labels, reg_param_r)
    final_gradient_norm = np.linalg.norm(gradient_k)
    print(f"最终回归系数: {beta_k}")
    print(f"最终目标函数值: {final_objective:.6f}")
    print(f"最终梯度范数: {final_gradient_norm:.6f}")
    print(f"总耗时: {total_time:.2f} 秒")
    


    return beta_k

def data(n, p, u):
    
    n1 = n // 2 
    n2 = n - n1 
    p_prime = p - 1 

   
    x_prime_part1 = np.random.randn(n1, p_prime)    
    x_prime_part2 = np.random.randn(n2, p_prime) + u
    x_prime_full = np.vstack((x_prime_part1, x_prime_part2))

    intercept_col = np.ones((n, 1))
    X_final = np.hstack((intercept_col, x_prime_full))

    y_labels_part1 = np.ones(n1, dtype=int)
    y_labels_part2 = np.zeros(n2, dtype=int)
    y_final = np.concatenate((y_labels_part1, y_labels_part2))

    return X_final, y_final


if __name__ == '__main__':
    n = 20000
    p = [11, 31, 101, 301]
    r = 0.01

    for p1 in p:
        print("p =", p1)
        p_ = p1 - 1
        u = np.full(p_, 0.1)

        x, y = data(n = n, p = p1, u = u)

        beta = np.zeros(p1)

        print("开始运行p =", p1, "时的BFGS优化")

        BFGS(feature_matrix=x, labels=y, reg_param_r=r, initial_beta_guess=beta, max_iterations=200)

        print()