
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

theta = 0
# refrance :https://medium.com/@bbssaivarma/implementation-of-logistic-regression-in-python-using-gradient-descent-without-sklearn-982ede00a3d2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#
# def gradient_descent(X, y, learning_rate, nbr_steps):
#     intercept = np.ones((X.shape[0], 1))
#     X = np.hstack((intercept, X))
#     weights = np.zeros(X.shape[1])
#     m = len(df)
#     for step in range(nbr_steps):
#         scores = np.dot(X, weights)
#         hypothesis = sigmoid(scores)
#         output_error_signal = hypothesis - y
#
#         gradient = np.dot(X.T, output_error_signal)
#
#         weights -= (learning_rate / m) * gradient
#
#     return weights

def Gradient_Descent_alg(X, y, theta, a, num_iters):
    length= len(y)
    for i in range(num_iters):
        h = sigmoid(X @ theta)
        theta = theta - (a / length) * (X.T @ (h - y))
    return theta


def error_test(x1, x2, y):
    error = theta[0] + theta[1] * x1 + theta[2] * x2
    error_rate = 0
    if error >= 0:
        error_rate = 1
    if error_rate == y:
        return 1
    else:
        return 0

def logistic_Regression(X, y, l_r, n_iters):

    X = np.c_[np.ones((len(X), 1)), X]
    theta = np.zeros(X.shape[1])
    theta = Gradient_Descent_alg(X, y, theta, l_r, n_iters)
    return theta

if __name__ == "__main__":
    data_train = pd.read_csv("train.csv")
    X = data_train[["x1", "x2"]].values
    Y = data_train["class"].values



    theta = logistic_Regression(X, Y, 0.01,20000)
    x1 = data_train["x1"].values
    x2 = data_train["x2"].values
    line_x1 = np.linspace(-1, 1, 100)
    x2 = -(theta[1] * line_x1 + theta[0]) / theta[2]
    print("the wight : " )
    print(theta)





    # Test Accuracy
    test = pd.read_csv("test.csv")
    X_test = test[["x1", "x2"]].values
    Y_test = test["class"].values



    x1_train = list(X[:, 0])
    x2_train = list(X[:, 1])
    y_train = list(Y)
    count = 0
    for i in range(len(x1_train)):
        count += error_test(x1_train[i], x2_train[i], y_train[i])
    accuracy = count / len(y_train)
    print("Train Accuracy cpmapre with test data: ", round(accuracy , 5))
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    # plt.scatter(X_t[:, 0], X_t[:, 1], c=Y_t)
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.plot(line_x1, x2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()



