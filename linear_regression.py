import numpy as np
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Contours, Scatter

from compute_cost import compute_cost


# ====================== Plot the data from data1 ======================

training_set = np.loadtxt('data1.txt', delimiter=',')
x = training_set[:,0]
y = training_set[:,1]

plot([Scatter(x=x, y=y, mode='markers')], filename='data1.html')


# ====================== Cost Function ======================

x = np.stack((np.ones(y.shape), x), axis=1)
X = np.matrix(x)
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

raw_input("Press Enter to continue\n")


# ====================== Gradient Descent ======================

