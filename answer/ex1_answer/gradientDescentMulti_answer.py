from computeCostMulti_answer import computeCostMulti
import numpy as np

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        for j in range(len(theta)):
                    theta[j] = theta[j] - (alpha/m)*np.sum((np.dot(X,theta) - y)*np.array(X[:,j]))

        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history