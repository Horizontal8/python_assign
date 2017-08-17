#COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
#   J = COSTFUNCTIONREG(theta, X, y, lambda_reg) computes the cost of using
#   theta as the parameter for regularized logistic regression and the
#   gradient of the cost w.r.t. to the parameters.

import numpy as np 
from sigmoid_answer import sigmoid

def costFunctionReg(theta, X, y, lambda_reg, return_grad=False):
	#initialize some useful values
	m = len(y) # number of training examples

	#You need to return the following variables correctly
	J = 0
	grad = np.zeros(theta.shape)

	# ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    # taken mostly from costFunction.py and added regularization term
    # note that we don't just take all of theta, but rather only n of the n+1 elements
    #	size(theta) is equal to [n+1  1], so we take the first element of that ( in size(theta,1) ) for
    #	the expression theta(2: size(theta, 1) )

	if return_grad == True:
		return J,grad
	else:
		return J

	# =============================================================
