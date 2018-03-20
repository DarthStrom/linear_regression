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

pause()


# ====================== Cost Function ======================

X = np.stack((np.ones(y.shape), x), axis=1)
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')

J = compute_cost(X, y, theta)
print('With theta = [[0], [0]]\nCost computed = %.2f\n' % J)

J = compute_cost(X, y, np.array([[-1], [2]]))
print('With theta = [[-1], [2]]\nCost computed = %.2f\n' % J)

pause()


# ====================== Gradient Descent ======================

print('\nRunning Gradient Descent ...\n')

theta = gradient_descent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent:\n')
print(theta)

J = compute_cost(X, y, theta)
print('Cost computed = %.2f\n' % J)

pause()

linear_fit = Scatter(x=X[:, 1], y=(X.dot(theta)).flatten())
plot([training_set_scatter, linear_fit], filename='linear-fit.html')

pause()

prediction = np.array([1, 14]).dot(theta)
print('For x=14, we predict y=%.2f\n' % prediction)
prediction = np.array([1, 42]).dot(theta)
print('For x=42, we predict y=%.2f\n' % prediction)

pause()


