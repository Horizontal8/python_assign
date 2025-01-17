## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.py
#     projectData.py
#     recoverData.py
#     computeCentroids.py
#     findClosestCentroids.py
#     kMeansInitCentroids.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import scipy.io 
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from featureNormalize import featureNormalize
from pca import pca 
from drawLine import drawLine
from projectData import projectData
from recoverData import recoverData
from displayData import displayData
from kMeansInitCentroids import kMeansInitCentroids
from runkMeans import runkMeans
from hsv import hsv

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize

print('Visualizing example dataset for PCA.')
#  The following command loads the dataset. You should now have the 
#  variable X in your environment
data = scipy.io.loadmat('ex7data1.mat')
X = data['X']

# interactive graphs
plt.ion()

#  Visualize the example dataset
plt.close()

#  Visualize the example dataset
plt.scatter(X[:, 0], X[:, 1], marker='o', color='b', facecolors='none', lw=1.0)
plt.axis([0.5, 6.5, 2, 8])
plt.axis('equal')
plt.show(block=False)

input('Program paused. Press <Enter> to continue...')

## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('Running PCA on example dataset.')

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S, V = pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
mu2 = mu + 1.5 * S.dot(U.T)
plt.hold(True)
plt.plot([mu[0], mu2[0, 0]], [mu[1], mu2[0, 1]], '-k', lw=2)
plt.plot([mu[0], mu2[1, 0]], [mu[1], mu2[1, 1]], '-k', lw=2)

print('Top eigenvector: ')
print(' U(:,1) = {:f} {:f}'.format(U[0,0], U[1,0]))
print('(you should expect to see -0.707107 -0.707107)')

input('Program paused. Press <Enter> to continue...')

## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('Dimension reduction on example dataset.')

#  Plot the normalized dataset (returned from pca)
plt.close()
plt.scatter(X_norm[:,0], X_norm[:,1], s=75, facecolors='none', edgecolors='b')
plt.axis([-4, 3, -4, 3])
plt.gca().set_aspect('equal', adjustable='box')
plt.show(block=False)

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: {:f}'.format(Z[0][0]))
print('(this value should be about 1.481274)')

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: {:f} {:f}'.format(X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)')

#  Draw lines connecting the projected points to the original points
plt.hold(True)
plt.scatter(X_rec[:, 0], X_rec[:, 1], s=75, facecolors='none', edgecolors='r')
for i in range(X_norm.shape[0]):
    drawLine(X_norm[i,:], X_rec[i,:], linestyle='--', color='k', linewidth=1)
plt.hold(False)

input('Program paused. Press <Enter> to continue...')

## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('Loading face dataset.\n');

#  Load Face dataset
mat = scipy.io.loadmat('ex7faces.mat')
X = np.array(mat["X"])

#  Display the first 100 faces in the dataset
displayData(X[:100, :])

input('Program paused. Press <Enter> to continue...')

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('Running PCA on face dataset.\n(this mght take a minute or two ...)\n')

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature
X_norm, _, _ = featureNormalize(X)

#  Run PCA
U, S, V = pca(X_norm)

#  Visualize the top 36 eigenvectors found
displayData(U[:, :36].T)

input('Program paused. Press <Enter> to continue...')

## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('Dimension reduction for face dataset.\n');

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print('{:d} {:d}'.format(Z.shape[0], Z.shape[1]))

input('Program paused. Press <Enter> to continue...')

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('Visualizing the projected (reduced dimension) faces.\n');

K = 100
X_rec  = recoverData(Z, U, K)

# Display normalized data
plt.close()
plt.subplot(1, 2, 1)
displayData(X_norm[:100,:])
plt.title('Original faces')
plt.gca().set_aspect('equal', adjustable='box')

# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
displayData(X_rec[:100,:])
plt.title('Recovered faces')
plt.gca().set_aspect('equal', adjustable='box')

input('Program paused. Press <Enter> to continue...')

## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

plt.close()

# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first

# A = double(imread('bird_small.png'));
mat = scipy.io.loadmat('bird_small.mat')
A = mat["A"]

# from ex7.py, part 4
A = A / 255.0
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3, order='F').copy()
K = 16 
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
#  use flatten(). otherwise, Z[sel, :] yields array w shape [1000,1,2]
sel = np.floor(np.random.rand(1000, 1) * X.shape[0]).astype(int).flatten()

#  Setup Color Palette
palette = hsv(K)
colors = np.array([palette[int(i)] for i in idx[sel]])

#  Visualize the data and centroid memberships in 3D
fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=100, c=colors)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show(block=False)

input('Program paused. Press <Enter> to continue...')

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, _, _ = featureNormalize(X)

# PCA and project the data to 2D
U, S, V = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
fig2 = plt.figure(2)
plt.scatter(Z[sel, 0], Z[sel, 1], s=100, c=colors)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
plt.show(block=False)

input('Program paused. Press <Enter> to continue...')