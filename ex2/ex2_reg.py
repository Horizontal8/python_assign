import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import fmin_bfgs
from plotData import plotData
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data2.txt',delimiter=",")
X = data[:,:2]
y = data[:,2]

print('Plotting data with + indicating (y=1) examples and o indicating (y=0) examples.')

plotData(X,y)
plt.legend(['y=1','y=0'],loc='upper right',shadow=True,numpoints=1)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()
input('Program paused. Press <Enter> to continue...')

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic 
#  regression to classify the data points. 
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:,0],X[:,1])
m,n = X.shape

# Initialize fitting parameters
initial_theta = np.zeros(n)

# Set regularization parameter lambda to 1
lambda_reg = 0

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg(initial_theta, X, y, lambda_reg)
print('Cost at initial theta (zeros): {:f}'.format(cost))

input('Program paused. Press <Enter> to continue...')

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and 
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?

# Initialize fitting parameters
initial_theta = np.zeros(n)

# Set regularization parameter lambda to 1 (you should vary this)
lambda_reg = 100

# Run fmin_bfgs to obtain the optimal theta
# This function returns theta and the cost
myargs = (X,y,lambda_reg)
result = fmin_bfgs(costFunctionReg,x0=initial_theta,args=myargs,full_output=True)
theta,cost = result[0],result[1]

print('lambda = ' + str(lambda_reg))
print('Cost at theta found by scipy: %f' % cost)
print('theta:', theta)

# Plot Boundary
plotDecisionBoundary(theta,X,y)
plt.legend(['y=1','y=0','Decision Boundary'],loc='upper right',shadow=True,numpoints=1)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('lambda = {:f}'.format(lambda_reg))
plt.show()

input('Program paused. Press <Enter> to continue...')

# Compute accuracy on our training set
p = predict(theta,X)
print('Train Accuracy: {:f}'.format(np.mean(p==y)*100))
input('Program paused. Press <Enter> to continue...')
