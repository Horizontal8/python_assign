## Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.py
#     dataset3Params.py
#     processEmail.py
#     emailFeatures.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io
from sklearn import svm
from processEmail import processEmail
from emailFeatures import emailFeatures
from getVocabList import getVocabList
## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

print('Preprocessing sample email (emailSample1.txt)')

# Extract Features
file = open('emailSample1.txt', 'r')
file_contents = file.readlines()
word_indices  = processEmail(''.join(file_contents))

# Print Stats
print('Word Indices: ')
print(word_indices)

input('Program paused. Press <Enter> to continue...')

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n.
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

print('Extracting features from sample email (emailSample1.txt)')

# Extract Features
file = open('emailSample1.txt')
file_contents = file.readlines()
word_indices = processEmail(''.join(file_contents))
features = emailFeatures(word_indices)

# Print Stats
print('Length of feature vector: %d'% features.size)
print('Number of non-zero entries: %d'% sum(features > 0))

input('Program paused. Press <Enter> to continue...')

## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
data = scipy.io.loadmat('spamTrain.mat')
X = data['X']
y = data['y']

print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...')

C = 0.1
clf = svm.SVC(C=C, kernel='linear', tol=1e-3, max_iter=200)
model = clf.fit(X, y)
train_predictions = clf.predict(X).reshape((y.shape[0],1))
train_acc = 100. * float(sum(train_predictions == y))/y.shape[0]
print('Training accuracy = %0.2f%%' % train_acc)

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
data = scipy.io.loadmat('spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest']

print('Evaluating the trained Linear SVM on a test set ...')
test_predictions = clf.predict(Xtest).reshape((ytest.shape[0],1))
test_acc = 100. * float(sum(test_predictions == ytest))/ytest.shape[0]
print('Test set accuracy = %0.2f%%' % test_acc)

## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtain the vocabulary list
w = model.coef_[0]

# from http://stackoverflow.com/a/16486305/583834
# reverse sorting by index
indices = w.argsort()[::-1][:15]
vocabList = sorted(getVocabList().keys())

print('\nTop predictors of spam: \n');
for idx in indices: 
    print( ' {:s} ({:f}) '.format( vocabList[idx], float(w[idx]) ) )

input('Program paused. Press <Enter> to continue...')

## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
#  The following code reads in one of these emails and then uses your
#  learned SVM classifier to determine whether the email is Spam or
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
filename = 'emailSample2.txt'

# Read and predict

file = open(filename)
file_contents = file.readlines()
word_indices = processEmail(''.join(file_contents))
x = emailFeatures(word_indices)
p = model.predict(x.flatten())

print('Processed %s\n\nSpam Classification: %d' % (filename, p))
print('(1 indicates spam, 0 indicates not spam)')