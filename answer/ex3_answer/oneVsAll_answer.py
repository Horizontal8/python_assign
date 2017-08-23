import numpy as np 
from scipy import optimize
from lrCostFunction_answer import lrCostFunction
from lrCostFunction_answer import lrGradient

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
	initial_theta = np.zeros((X.shape[1],1)).reshape(-1)

	for c in range(num_labels):
		cclass = c+1
		print('Training {:d} out of {:d} categories...'.format(c+1,num_labels))
		logic_Y = np.array([1 if x == cclass else 0 for x in y])
		myargs = (X, logic_Y, lambda_reg)
		result = optimize.fmin_cg(lrCostFunction, fprime=lrGradient, x0=initial_theta, args=myargs, maxiter=50, disp=False, full_output=True)
		all_theta[c,:] = result[0]

	# =========================================================================
	return all_theta