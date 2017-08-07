import matplotlib.pyplot as plt 

def plotData(X,y):
	"""
    plots the data points and gives the figure axes labels of
    population and profit.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Plot the training data into a figure using the
#               "figure" and "scatter" commands. Set the axes labels using
#               the "xlabel" and "ylabel" commands. Assume the
#               population and revenue data have been passed in
#               as the x and y arguments of this function.
#
# Hint: You can use the 'marker' and color option with plot to have 
#       the markers appear as red crosses. 


	plt.figure() # open a new figure window
	plt.scatter(X,y,marker='x',color='r',label='Training data')
	plt.ylabel('Profit in $10,000s')
	plt.xlabel('Population of City in 10,000s')
# ============================================================
