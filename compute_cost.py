import numpy as np

def compute_cost(X, y, theta):

    m = y.size

    error = (theta.transpose() * X.transpose() - y)

    return error * error.transpose() / (2*m)
