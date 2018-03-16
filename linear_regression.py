import numpy as np
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Contours, Scatter

from compute_cost import compute_cost
from gradient_descent import gradient_descent
from pause import pause


# ====================== Plot the data from data1 ======================

training_set = np.loadtxt('data1.txt', delimiter=',')
x = training_set[:, 0]
y = training_set[:, 1]

training_set_scatter = Scatter(x=x, y=y, mode='markers')
plot([training_set_scatter], filename='data1.html')


# ====================== Cost Function ======================

X = np.stack((np.ones(y.shape), x), axis=1)
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')

J = compute_cost(X, y, theta)
print('With theta = [[0], [0]]\nCost computed = %.2f\n' % J)
print('Expected cost value: 32.07\n')

J = compute_cost(X, y, np.array([[-1], [2]]))
print('With theta = [[-1], [2]]\nCost computed = %.2f\n' % J)
print('Expected cost value: 54.24\n')

pause()


# ====================== Gradient Descent ======================

print('\nRunning Gradient Descent ...\n')

theta = gradient_descent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent:\n')
print(theta)
print('\nExpected theta values: [[-3.6303], [1.1664]]\n\n')

pause()

linear_fit = Scatter(x=X[:, 1], y=(X.dot(theta)).flatten())
plot([training_set_scatter, linear_fit], filename='linear-fit.html')
