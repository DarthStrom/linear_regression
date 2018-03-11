import numpy as np
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Contours, Scatter

# Plot the data from data1
training_set = np.loadtxt('data1.txt', delimiter=',')
xs = training_set[:,0]
ys = training_set[:,1]

plot([Scatter(x=xs, y=ys, mode='markers')], filename='data1.html')
