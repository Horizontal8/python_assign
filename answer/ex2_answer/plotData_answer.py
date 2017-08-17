import numpy as np 
from matplotlib import pyplot as plt 


def plotData(X,y):

# ====================== YOUR CODE HERE ======================
	pos = X[np.where(y==1,True,False)]
	neg = X[np.where(y==0,True,False)]
	plt.plot(pos[:,0],pos[:,1],'+',markersize=7,markeredgecolor='black',markeredgewidth=2)
	plt.plot(neg[:,0],neg[:,1],'o',markersize=7,markeredgecolor='black',markerfacecolor='yellow')


# =========================================================================