import numpy as np
np.random.seed(42)

n = 1000
p = 10000
k = 50
lambda_param = 0.05

X = np.random.normal(size=(n, p))

beta = np.zeros(p)
v = np.random.normal(size=(0, 1, k))
beta[:k] = v ** 2

epsilon = np.random.normal(size=(0, 1, n))

y = X @ beta + epsilon

