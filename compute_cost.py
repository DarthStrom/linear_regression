import numpy as np

def compute_cost(X, y, theta):

    m = y.size

    error = (theta.transpose().dot(X.transpose()) - y)

    return error.dot(error.transpose()) / (2*m)
