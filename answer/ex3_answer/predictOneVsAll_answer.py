from scipy.special import expit #Vectorized sigmoid function
import numpy as np

def h(mytheta,myX): #Logistic hypothesis function
    return expit(np.dot(myX,mytheta))

def predictOneVsAll(all_theta,X):
#PREDICT Predict the label for a trained one-vs-all classifier. The labels 
#are in the range 1..K, where K = size(all_theta, 1). 
#  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
#  for each example in the matrix X. Note that X contains the examples in
#  rows. all_theta is a matrix where the i-th row is a trained logistic
#  regression theta vector for the i-th class. You should set p to a vector
#  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
#  for 4 examples) 
	m = X.shape[0]
	num_labels = all_theta.shape[0]

	# You need to return the following varibles correctly
	p = np.zeros((m,1))

	# Add ones to the X data matrix
	X = np.column_stack((np.ones((m,1)),X))

	# ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters (one-vs-all).
    #               You should set p to a vector of predictions (from 1 to
    #               num_labels).
    #  

	p_adj = np.zeros((m,1))

	p_adj = np.argmax(h(all_theta.T,X),axis=1)

	# offsets python's zero notation

	p = [i+1 for i in p_adj]
	# =========================================================================

	return p