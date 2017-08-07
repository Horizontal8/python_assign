import numpy as np

def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """
    X_norm, mu, sigma = X.copy(),[],[]
    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    # Hint: You might find the 'mean' and 'std' functions useful.
    #
    for i in range(X.shape[1]):
        mu+=[np.mean(X[:,i])]
        sigma+=[np.std(X[:,i])]
        X_norm[:,i]=(X_norm[:,i]-mu[-1])/sigma[-1]

# ============================================================

    return X_norm, mu, sigma