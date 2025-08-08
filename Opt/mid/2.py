import numpy as np

def sigmoid(z):
    # 计算sigmoid函数
    return 1 / (1 + np.exp(-z))

def compute_loss(w, X, y, lam):
    # 计算带L2正则化的Logistic回归损失
    n = len(y)
    pred = sigmoid(np.dot(X, w))
    # 避免log(0)出现，做简单的数值稳定处理
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    loss = - (1/n) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    reg = (lam / 2) * np.sum(w * w)
    return loss + reg

def compute_gradient(w, X, y, lam):
    # 计算带L2正则化的梯度
    n = len(y)
    pred = sigmoid(np.dot(X, w))
    grad = (1/n) * np.dot(X.T, (pred - y)) + lam * w
    return grad

def bfgs_logistic_regression(X, y, lam=0.1, max_iter=50, tol=1e-5):
    n_samples, n_features = X.shape
    # 初始化权重为0
    w = np.zeros(n_features)
    # 初始化Hessian矩阵的逆矩阵为单位矩阵
    H = np.eye(n_features)
    
    for i in range(max_iter):
        grad = compute_gradient(w, X, y, lam)
        
        # 如果梯度很小，就停止迭代
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print("梯度很小，停止迭代，迭代次数：", i)
            break
        
        # 计算搜索方向 d = -H * grad
        d = -np.dot(H, grad)
        
        # 简单的固定步长线搜索（不使用复杂的Armijo条件）
        alpha = 1.0
        # 这里做个简单的尝试，最多缩小5次步长
        for _ in range(5):
            w_new = w + alpha * d
            loss_new = compute_loss(w_new, X, y, lam)
            loss_old = compute_loss(w, X, y, lam)
            if loss_new < loss_old:
                break
            alpha = alpha * 0.5
        
        # 更新参数
        s = w_new - w
        grad_new = compute_gradient(w_new, X, y, lam)
        y_vec = grad_new - grad
        
        # 更新H矩阵，避免除0错误，简单判断一下
        if np.dot(y_vec, s) > 1e-10:
            rho = 1.0 / np.dot(y_vec, s)
            I = np.eye(n_features)
            H = (I - rho * np.outer(s, y_vec)).dot(H).dot(I - rho * np.outer(y_vec, s)) + rho * np.outer(s, s)
        
        w = w_new
        
        print("第{}次迭代，损失值：{:.6f}，梯度范数：{:.6f}".format(i+1, loss_new, grad_norm))
    
    return w

# 下面是一个简单的测试例子
if __name__ == "__main__":
    # 生成简单的二分类数据
    np.random.seed(0)
    n_samples = 100
    n_features = 2
    
    X_pos = np.random.randn(n_samples//2, n_features) + 1
    X_neg = np.random.randn(n_samples//2, n_features) - 1
    X = np.vstack((X_pos, X_neg))
    y = np.array([1]*(n_samples//2) + [0]*(n_samples//2))
    
    # 添加偏置项
    X = np.hstack((np.ones((n_samples, 1)), X))
    
    # 调用BFGS算法训练
    w_opt = bfgs_logistic_regression(X, y, lam=0.1, max_iter=50)
    
    print("训练完成，最终权重：", w_opt)
    
    # 简单预测准确率
    preds = sigmoid(np.dot(X, w_opt)) > 0.5
    accuracy = np.mean(preds == y)
    print("训练准确率：{:.2f}%".format(accuracy * 100))
