import numpy as np
from numpy.linalg import inv

def normalEqn(X,y):
    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """
    theta = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#

# ---------------------- Sample Solution ----------------------


# -------------------------------------------------------------
    theta = np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)

    return theta

# ============================================================
