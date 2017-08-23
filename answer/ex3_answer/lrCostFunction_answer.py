#LRCOSTFUNCTION and LRGRADIENT Compute cost and gradient for logistic regression with 
#regularization
#   J = LRCOSTFUNCTION(theta, X, y, lambda_reg) computes the cost of using
#   theta as the parameter for regularized logistic regression
#   grad = LRGRADIENT(theta, X, y, lambda_reg) computes the
#   gradient of the cost w.r.t. the parameters. 

import numpy as np
from scipy.special import expit #Vectorized sigmoid function

def h(mytheta,myX): #Logistic hypothesis function
    return expit(np.dot(myX,mytheta))

def lrCostFunction(mytheta,myX,myy,mylambda = 0.):
	# ====================== YOUR CODE HERE ======================
    m = myX.shape[0] #5000
    myh = h(mytheta,myX) #shape: (5000,1)
    term1 = np.log( myh ).dot( -myy.T ) #shape: (5000,5000)
    term2 = np.log( 1.0 - myh ).dot( 1 - myy.T ) #shape: (5000,5000)
    left_hand = (term1 - term2) / m #shape: (5000,5000)
    right_hand = mytheta.T.dot( mytheta ) * mylambda / (2*m) #shape: (1,1)
    J = left_hand + right_hand
    # =============================================================

    return J #shape: (5000,5000)

def lrGradient(mytheta,myX,myy,mylambda = 0.):
	# ====================== YOUR CODE HERE ======================
    m = myX.shape[0]
    #Tranpose y here because it makes the units work out in dot products later

    beta = h(mytheta,myX)-myy.T #shape: (5000,5000)

    #regularization skips the first element in theta
    regterm = mytheta[1:]*(mylambda/m) #shape: (400,1)

    grad = (1./m)*np.dot(myX.T,beta) #shape: (401, 5000)
    #regularization skips the first element in theta
    grad[1:] = grad[1:] + regterm
    # =============================================================
    
    return grad #shape: (401, 5000)