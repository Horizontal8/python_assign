#   PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
#   the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
#   positive examples and o for the negative examples. X is assumed to be 
#   a either 
#   1) Mx3 matrix, where the first column is an all-ones column for the 
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones

import matplotlib.pyplot as plt 
import numpy as np 
from mapFeature import mapFeature 
from plotData import plotData 

def plotDecisionBoundary(theta,X,y):
	plt.figure()
	plotData(X[:,1:],y)

	if X.shape[1]<=3:
		#Only need 2 points to define a line, so choose two endpoints
		plot_x = np.array([min(X[:,2]),max(X[:,2])])

		#Calculate the decision boundary line
		#theta0 + theta1*x1 + theta2*x2 = 0
        #y=mx+b is replaced by x2 = (-1/thetheta2)(theta0 + theta1*x1)
		plot_y = (-1/theta[2])*(theta[1]*plot_x+theta[0])
		plt.plot(plot_x,plot_y)
	else:
		u = np.linspace(-1,1.5,50)
		v = np.linspace(-1,1.5,50)
		z = np.zeros((len(u),len(v)))
		for i in range(len(u)):
			for j in range(len(v)):
				z[i,j] = np.dot(mapFeature(np.array([u[i]]),np.array([v[j]])),theta)
		z = np.transpose(z)
		plt.contour(u, v, z, levels=[0], linewidth=2).collections[0]
