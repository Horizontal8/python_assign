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
    



    # =============================================================

    return J #shape: (5000,5000)

def lrGradient(mytheta,myX,myy,mylambda = 0.):
	# ====================== YOUR CODE HERE ======================
    


    
    # =============================================================
    
    return grad #shape: (401, 5000)