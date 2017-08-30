## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.py
#     learningCurve.py
#     validationCurve.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import scipy.io 
import matplotlib.pyplot as plt
import numpy as np

from linearRegCostFunction_answer import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
from validationCurve import validationCurve

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = scipy.io.loadmat('ex5data1.mat')

# m = Number of examples
X = data['X'][:, 0]
y = data['y'][:, 0]
Xval = data['Xval'][:, 0]
yval = data['yval'][:, 0]
Xtest = data['Xtest'][:, 0]
ytest = data["ytest"][:, 0]
m = X.size

# Plot training data
plt.scatter(X, y, marker='x', s=60, color='r', lw=1.5)
plt.ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
plt.xlabel('Change in water level (x)')     # Set the x-axis label
plt.show()

input('Program paused. Press <Enter> to continue...')

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.array([1, 1])
J = linearRegCostFunction(np.column_stack((np.ones(m), X)), y, theta, 1)

print('Cost at theta = [1  1]: %f \n(this value should be about 303.993192)\n' % J)

input('Program paused. Press <Enter> to continue...')

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

theta = np.array([1, 1])
J, grad = linearRegCostFunction(np.column_stack((np.ones(m), X)), y, theta, 1, True)

print('Gradient at theta = [1  1]:  [%f %f] \n(this value should be about [-15.303016 598.250744])\n' %(grad[0], grad[1]))

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with Lambda = 0
Lambda = 0
theta = trainLinearReg(np.column_stack((np.ones(m), X)), y, 1)

#  Plot fit over the data
plt.scatter(X, y, marker='x', s=20, color='r', lw=1.5)
plt.ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
plt.xlabel('Change in water level (x)')     # Set the x-axis label
plt.plot(X, np.column_stack((np.ones(m), X)).dot(theta), '--', lw=2.0)
plt.show()

input('Program paused. Press <Enter> to continue...')

## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias"
#

Lambda = 0
error_train, error_val = learningCurve(np.column_stack((np.ones(m), X)), y,
                                       np.column_stack((np.ones(Xval.shape[0]), Xval)), yval, Lambda)
plt.figure()
plt.plot(range(m), error_train, color='b', lw=0.5, label='Train')
plt.plot(range(m), error_val, color='r', lw=0.5, label='Cross Validation')
plt.title('Learning curve for linear regression')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')

plt.xlim(0, 13)
plt.ylim(0, 150)
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
plt.show()

print('Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

input('Program paused. Press <Enter> to continue...')

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 5

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.column_stack((np.ones(m), X_poly))                   # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_test))        # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_val))           # Add Ones

print('Normalized Training Example 1:')
print(X_poly[0, :])

input('Program paused. Press <Enter> to continue...')

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of Lambda. The code below runs polynomial regression with 
#  Lambda = 0. You should try running the code with different values of
#  Lambda to see how the fit and learning curve change.
#

Lambda = 0
theta = trainLinearReg(X_poly, y, Lambda)

# Plot training data and fit
plt.figure()
plt.scatter(X, y, marker='x', s=10, color='r', lw=1.5)

plotFit(min(X), max(X), mu, sigma, theta, p)

plt.xlabel('Change in water level (x)')            # Set the y-axis label
plt.ylabel('Water flowing out of the dam (y)')     # Set the x-axis label
# plt.plot(X, np.column_stack((np.ones(m), X)).dot(theta), marker='_',  lw=2.0)
plt.title('Polynomial Regression Fit (Lambda = %f)' % Lambda)
plt.show()

error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, Lambda)
plt.plot(range(m), error_train, label='Train')
plt.plot(range(m), error_val, label='Cross Validation')
plt.title('Polynomial Regression Learning Curve (Lambda = %f)' % Lambda)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.legend()
plt.show()

print('Polynomial Regression (Lambda = %f)\n\n' % Lambda)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

input('Program paused. Press <Enter> to continue...')

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  Lambda on a validation set. You will then use this to select the
#  "best" Lambda value.
#
p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.column_stack((np.ones(m), X_poly))                   # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_test))        # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_val))           # Add Ones
Lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

plt.plot(Lambda_vec, error_train, label='Train')
plt.plot(Lambda_vec, error_val, label='Cross Validation')
plt.legend(loc='upper left', shadow=True, fontsize='x-large', numpoints=1)
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.show()

print('Lambda\t\tTrain Error\tValidation Error')
for i in range(Lambda_vec.size):
    print(' %f\t%f\t%f' % (Lambda_vec[i], error_train[i], error_val[i]))

input('Program paused. Press <Enter> to continue...')

## =========== Part 9: Computing test set error on the best lambda found =============
#

# best lambda value from previous step
lambda_val = 3

# note that we're using X_poly - polynomial linear regression with polynomial features
theta = trainLinearReg(X_poly, y, lambda_val)

# because we're using X_poly, we also have to use X_poly_test with polynomial features
error_test = linearRegCostFunction(X_poly_test, ytest, theta, 0)
print('Test set error: {:f}\n'.format(error_test)) # expected 3.859

input('Program paused. Press <Enter> to continue...')

## =========== Part 10: Plot learning curves with randomly selected examples =============
#

# lambda_val value for this step
lambda_val = 0.01

# number of iterations
times = 50

# initialize error matrices
error_train_rand = np.zeros((m, times))
error_val_rand   = np.zeros((m, times))

for i in range(1,m+1):

    for k in range(times):

        # choose i random training examples
        rand_sample_train = np.random.permutation(X_poly.shape[0])
        rand_sample_train = rand_sample_train[:i]

        # choose i random cross validation examples
        rand_sample_val   = np.random.permutation(X_poly_val.shape[0])
        rand_sample_val   = rand_sample_val[:i]

        # define training and cross validation sets for this loop
        X_poly_train_rand = X_poly[rand_sample_train,:]
        y_train_rand      = y[rand_sample_train]
        X_poly_val_rand   = X_poly_val[rand_sample_val,:]
        yval_rand         = yval[rand_sample_val]             

        # note that we're using X_poly_train_rand and y_train_rand in training
        theta = trainLinearReg(X_poly_train_rand, y_train_rand, lambda_val)
            
        # we use X_poly_train_rand, y_train_rand, X_poly_train_rand, X_poly_val_rand
        error_train_rand[i-1,k] = linearRegCostFunction(X_poly_train_rand, y_train_rand, theta, 0)
        error_val_rand[i-1,k]   = linearRegCostFunction(X_poly_val_rand,   yval_rand,    theta, 0)


error_train = np.mean(error_train_rand, axis=1)
error_val   = np.mean(error_val_rand, axis=1)

# resets plot 
plt.close()

p1, p2 = plt.plot(range(m), error_train, range(m), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(lambda_val))
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show()


print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, error_train[i], error_val[i]))