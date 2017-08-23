import numpy as np 
from scipy import optimize
from lrCostFunction import lrCostFunction
from lrCostFunction import lrGradient

def oneVsAll(X,y,num_labels,lambda_reg):
	#ONEVSALL trains multiple logistic regression classifiers and returns all
	#the classifiers in a matrix all_theta, where the i-th row of all_theta
	#corresponds to the classifier for label i

	# Some useful variables
	m, n = X.shape

	# You need to return the following variables correctly
	all_theta = np.zeros((num_labels,n+1))

	# Add ones to the X data matrix
	X = np.column_stack((np.ones((m,1)),X))

	# ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda. 
    #
	


	

	# =========================================================================
	return all_theta