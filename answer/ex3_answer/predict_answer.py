import numpy as np 
from scipy.special import expit #Vectorized sigmoid function

def h(mytheta,myX): #Logistic hypothesis function
    return expit(np.dot(myX,mytheta))

def predict(Theta1, Theta2, X):
	#PREDICT Predict the label of an input given a trained neural network
    # p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    # trained weights of a neural network (Theta1, Theta2)
	# Useful values
	m = X.shape[0]
	num_labels = Theta2.shape[0]

	# You need to return the following variables correctly
	p = np.zeros((m,1))

	# ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a 
    #               vector containing labels between 1 to num_labels.
    #

    # add column of ones as bias unit from input layer to second layer
	X = np.column_stack((np.ones((m,1)),X)) # = a1
	# calculate second layer as h(z2) where z2=Theta1*a1
	a2 = h(Theta1.T,X)
	# add column of ones as bias unit from second layer to third layer
	a2 = np.column_stack((np.ones((a2.shape[0],1)),a2))
	# calculate third layer as h(z3) where z3=Theta2*a2
	a3 = h(Theta2.T,a2)
	# get indices as in predictOneVsAll
	p_adj = np.zeros((m,1))
	p_adj = np.argmax(a3,axis=1)
	# offsets python's zero notation
	p = [i+1 for i in p_adj]

	# =========================================================================

	return p 
