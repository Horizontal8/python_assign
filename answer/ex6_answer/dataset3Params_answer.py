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
    #

    ### determining best C and sigma

    best_pair, best_score = (0, 0), 0

    # iterate over values of sigma and C
    for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:

            # train model on training corpus with current sigma and C
            gamma = 1.0 / (2.0 * sigma ** 2)
            clf = svm.SVC(C=C, kernel='rbf', tol=1e-3, max_iter=200, gamma=gamma)
            model = clf.fit(X, y)

            this_score = clf.score(Xval,yval)
            if this_score>best_score:
                best_score = this_score
                best_pair = (C,sigma)

    C     = best_pair[0]
    sigma = best_pair[1]
    
    # =========================================================================
    return C, sigma