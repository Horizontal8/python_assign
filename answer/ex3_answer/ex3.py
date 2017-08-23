## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
import scipy.io 
import numpy as np
from displayData import displayData
from oneVsAll_answer import oneVsAll
from predictOneVsAll_answer import predictOneVsAll

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     lrCostFunction.py (logistic regression cost function)
#     oneVsAll.py
#     predictOneVsAll.py
#     predict.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10         # 10 labels, from 1 to 10

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = scipy.io.loadmat('ex3data1.mat') #training data stored in arrays X,y
X = data['X']
y = data['y']
m, _ = X.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[:100],:]

displayData(sel)

input('Program paused. Press <Enter> to continue...')

## ============ Part 2: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

print('Training One-vs-All Logistic Regression...')

lambda_reg = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_reg)
input('Program paused. Press <Enter> to continue...')

## ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X)
accuracy = sum(1 if a==b else 0 for a,b in zip(pred,y))/len(pred) * 100
print('\nTraining Set Accuracy: %0.01f%%\n' % accuracy)