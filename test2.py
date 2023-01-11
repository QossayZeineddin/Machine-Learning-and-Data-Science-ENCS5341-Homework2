import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


theta = 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_descent(X, y, theta, alpha, num_iters):
    length = len(y)
    for i in range(num_iters):
        h = sigmoid(X @ theta)
        theta = theta - (alpha / length) * (X.T @ (h - y))
    return theta

def logistic_regression(X, y, learning_rate, num_iters):

    theta = np.zeros(X.shape[1])
    theta = gradient_descent(X, y, theta, learning_rate, num_iters)
    return theta

def error_test(x1, x2, y):
    error = theta[0] + theta[1] * x1**2 + theta[2] * x2**2
    error_rate = 0
    if error >= 0:
        error_rate = 1
    if error_rate == y:
        return 1
    else:
        return 0

# refrance :https://medium.com/@bbssaivarma/implementation-of-logistic-regression-in-python-using-gradient-descent-without-sklearn-982ede00a3d2

if __name__ == "__main__":
    data_train = pd.read_csv("train.csv")
    x = data_train[["x1", "x2"]].values
    y = data_train["class"].values
    # praper the data
    xx = x
    yy = y
    y = np.array(list(y))
    x = np.c_[np.ones((len(x), 1)), np.square(x)]
    theta = logistic_regression(x, y, 0.01, 200000)
    print("the wight : ")
    print(theta)
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)
    x1, x2 = np.meshgrid(x1, x2)
    circle = theta[0] + theta[1] * x1 ** 2 + theta[2] * x2 ** 2

    plt.scatter(xx[:, 0], xx[:, 1], c=y)
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.contour(x1, x2, circle, [0])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


    # Test Dataset

    test_data = pd.read_csv("test.csv")
    X_test = test_data[["x1", "x2"]].values
    Y_test = test_data["class"].values
    Y_t = list(Y_test)
    x1_t = list(X_test[:, 0])
    x2_t = list(X_test[:, 1])
    y_t = list(Y_t)
    count = 0
    for i in range(len(x1_t)):
        count += error_test(x1_t[i], x2_t[i], y_t[i])
    accuracy = count / len(y_t)
    x1_train = list(xx[:, 0])
    x2_train = list(xx[:, 1])
    y_train = list(y)
    count = 0
    for i in range(len(x1_train)):
        count += error_test(x1_train[i], x2_train[i], y_train[i])
    accuracy = count / len(y_train)
    print("Train Accuracy cpmapre with test data: ", round(accuracy , 5))

