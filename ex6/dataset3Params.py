import numpy as np
from sklearn import svm

def dataset3Params(X, y, Xval, yval):
    #EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
    #where you select the optimal (C, sigma) learning parameters to use for SVM
    #with RBF kernel
    #   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
    #   sigma. You should complete this function to return the optimal C and 
    #   sigma based on a cross-validation set.
    #

    # You need to return the following variables correctly.
    sigma = 0.3
    C = 1

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You may use the 'score' method after training SVM model to 
    #               calculate error rate of the classification.
    #

    
    # =========================================================================
    return C, sigma