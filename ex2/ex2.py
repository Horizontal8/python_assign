# Logistic Regression
import numpy as np 
import matplotlib.pyplot as plt  

from plotData import plotData
from sigmoid import sigmoid
from costFunction import costFunction
from scipy.optimize import fmin
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
#     sigmoid.py
#     costFunction.py
#     gradientFunction.py
#     predict.py
#     costFunctionReg.py
#     gradientFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data1.txt',delimiter=',')
X = data[:,0:2]
y = data[:,2]

# ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y=1) examples and o indicating (y=0) examples.')

plotData(X,y)
plt.legend(['Admitted','Not admitted'],loc='upper right',shadow=True,numpoints=1)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()
input('Program paused. Press <Enter> to continue...')

## ============ Part 2: Compute Cost and Gradient ============
##  Setup the data matrix appropriately, and add ones for the intercept term
m,n = X.shape
# Add intercept term to x and X_test
X = np.concatenate((np.ones((m,1)),X),axis=1)
# Initialize fitting parameters
initial_theta = np.zeros(n+1)
# Compute and display initial cost and gradient
cost,grad = costFunction(initial_theta,X,y,return_grad=True)
print('Cost at initial theta (zeros): %f' % cost)
print('Gradient at initial theta (zeros):'+str(grad))
input('Program paused. Press <Enter> to continue...')

## ============= Part 3: Optimizing using fmin (and fmin_bfgs)  =============
#  In this exercise, you will use a built-in function (fmin) to find the
#  optimal parameters theta.

#  Run fmin to obtain the optimal theta
#  This function will return theta and the cost 

myargs = (X,y)
result = fmin(costFunction,x0=initial_theta,args=myargs,full_output=True)
theta,cost = result[0],result[1]

# Print theta to screen
print('Cost at theta found by scipy: %f' % cost)
print('theta:',theta)

# Plot Boundary
plotDecisionBoundary(theta,X,y)

# Labels and Legend
plt.legend(['Admitted', 'Not admitted','Decision Boundary'], loc='upper right', shadow=True, numpoints=1)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()

input('Program paused. Press <Enter> to continue...')


#  ============== Part 4: Predict and Accuracies ==============

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid(np.array([1,45,85]).dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of %f'%prob)

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy:{:f}'.format(np.mean(p==y)*100))
