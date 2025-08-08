import numpy as np
from scipy.optimize import minimize

# set some variables
n = 20000  # number of samples
p = 11     # number of features
r = 0.01   # regularization thing

# make empty arrays for data
X = np.zeros((n, p))
y = np.zeros(n)

# first half of data where y=1
for i in range(10000):
    x_temp = np.random.normal(0, 1, p-1)  # random normal stuff
    X[i, 0] = 1                           # intercept is 1
    X[i, 1:] = x_temp                     # rest of features
    y[i] = 1                              # label is 1

# second half where y=0
u = 0.1 * np.ones(p-1)  # mean vector all 0.1
for i in range(10000, 20000):
    x_temp = np.random.normal(u, 1, p-1)  # shift by u
    X[i, 0] = 1                           # intercept again
    X[i, 1:] = x_temp
    y[i] = 0                              # label is 0

# sigmoid function i guess
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# the function we need to minimize
def objective(beta):
    reg = r * np.sum(beta**2)  # regularization part
    total = reg
    for i in range(n):
        eta = np.dot(X[i], beta)  # dot product
        s = sigmoid(eta)
        if y[i] == 1:
            total -= np.log(s)    # log loss for y=1
        else:
            total -= np.log(1 - s)  # log loss for y=0
    return total

# gradient function
def gradient(beta):
    grad = 2 * r * beta  # gradient of regularization
    for i in range(n):
        eta = np.dot(X[i], beta)
        s = sigmoid(eta)
        grad += (s - y[i]) * X[i]  # add gradient of loss
    return grad

# starting beta with zeros
beta_start = np.zeros(p)

# run the bfgs thing
result = minimize(objective, beta_start, method='BFGS', jac=gradient)

# show some results
print("beta we got:", result.x)
print("final objective:", result.fun)

# check accuracy
correct = 0
for i in range(n):
    eta = np.dot(X[i], result.x)
    prob = sigmoid(eta)
    if prob > 0.5:
        pred = 1
    else:
        pred = 0
    if pred == y[i]:
        correct += 1
accuracy = correct / n
print("accuracy on training:", accuracy)

# double check the objective
check_obj = objective(result.x)
print("objective again:", check_obj)

# see if gradient is small
final_grad = gradient(result.x)
print("gradient at end:", final_grad)
print("gradient size:", np.linalg.norm(final_grad))