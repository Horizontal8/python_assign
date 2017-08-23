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

    


    

	# =========================================================================

	return p 
