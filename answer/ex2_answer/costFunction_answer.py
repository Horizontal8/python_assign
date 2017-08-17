import numpy as np 
from sigmoid_answer import sigmoid

def costFunction(theta,X,y,return_grad=False):
	# Initialize some useful values
	m = len(y) # number of training examples

	# You need to return the following variables correctly
	J = 0
	grad = np.zeros(theta.shape)
	# ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #

	term1 = y*np.transpose(np.log(sigmoid(np.dot(X,theta))))
	term2 = (1-y)*np.transpose(np.log(1-sigmoid(np.dot(X,theta))))
	J = -(1/m)*(term1+term2).sum()
	grad = (1/m)*np.dot(sigmoid(np.dot(X,theta)).T-y,X).T

	if return_grad==True:
		return J,np.transpose(grad)
	elif return_grad==False:
		return J
	# =============================================================

	


