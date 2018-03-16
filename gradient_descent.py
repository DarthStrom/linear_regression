import numpy as np

def gradient_descent(X, y, theta, alpha, iterations):

    m = y.size

    for _ in range(1, iterations):
        error = theta.transpose().dot(X.transpose()) - y
        theta = theta - alpha * X.transpose().dot(error.transpose()) / m

    return theta
