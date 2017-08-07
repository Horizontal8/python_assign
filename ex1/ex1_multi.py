#  Machine Learning Online Class
# Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression exercise. 
#
#  You will need to complete the following functions in this 
#  exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

# Initialization

# ================ Part 1: Feature Normalization ================

print('Loading data ...')

# Load Data
data = np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:,:2]
y = data[:,2]
m = y.T.size

# Print out some data points
print('First 10 examples from the dataset:')
print(np.column_stack((X[:10],y[:10])))

input('Program paused. Press <Enter> to continue...')

# Scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = featureNormalize(X)
print('[mu] [sigma]')
print(mu, sigma)


#  Add intercept term to X
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha). 
#
#               Your task is to first make sure that your functions - 
#               computeCost and gradientDescent already work with 
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with 
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(J_history, '-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
input('Program paused. Press <Enter> to continue...')

# Display gradient descent's result
print('Theta computed from gradient descent:')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
price = 0 # You should change this


# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house')
print('(using gradient descent):\n $%f\n' % price)

input('Program paused. Press <Enter> to continue...')

# ================ Part 3: Normal Equations ================

print('Solving with normal equations...')

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form 
#               solution for linear regression using the normal
#               equations. You should complete the code in 
#               normalEqn.m
#
#               After doing so, you should complete this code 
#               to predict the price of a 1650 sq-ft, 3 br house.
#

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.T.size

# Add intercept term to X
X = np.concatenate((np.ones((m,1)), X), axis=1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print(' %s \n' % theta)


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = 0 # You should change this


# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house')
print('(using normal equations):\n $%f\n' % price)

input('Program paused. Press <Enter> to continue...')