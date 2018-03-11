import numpy as np
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *

x = np.random.randn(2000)
y = np.random.randn(2000)

plot([Histogram2dContour(x=x, y=y, contours=Contours(coloring='heatmap'))],
      show_link=False,
      filename='otherplot.html')

