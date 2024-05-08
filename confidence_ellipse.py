from numpy import linalg as LA
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt
import numpy as np
import math

# inputs:
# mu    = mean vector
# Sigma = covariance matrix
# alpha = level
# color = plot color
def confidence_ellipse( mu, Sigma, alpha, color ):

    # Sanity checks
    if not np.allclose(Sigma, Sigma.T, rtol=1e-05, atol=1e-08):
        raise Exception("Covariance matrix is not symmetric")

    # Sigma must be symmetric and PSD
    if not np.all(LA.eigvals(Sigma) > 0) or not np.all(np.isreal(LA.eigvals(Sigma))):
        raise Exception("Covariance matrix is not positive definite")

    # alpha must be between 0 and 1
    if alpha <= 0 or alpha >= 1:
        raise Exception("Alpha should be strictly between 0 and 1")

    # Set up
    z = chi2.ppf(alpha, df=2)


    # eigenvalues, eigenvectors
    eigenvalues, eigenvectors = LA.eig( Sigma )
    lambda1 = eigenvalues[0]
    lambda2 = eigenvalues[1]
    theta0  = math.atan2( eigenvectors[1,0], eigenvectors[0,0]  )

    # Plot
    N = 50
    theta = np.linspace(0, 2*np.pi, num=N)

    # use polar coordinates to generate points on the ellipse
    x = mu[0] + np.sqrt( lambda1 * z  )*np.cos(theta)*np.cos(theta0) - np.sqrt(lambda2*z)*np.sin(theta)*np.sin(theta0)
    y = mu[1] + np.sqrt( lambda1 * z  )*np.cos(theta)*np.sin(theta0) + np.sqrt(lambda2*z)*np.sin(theta)*np.cos(theta0)

    plt.plot( x, y, color  )
    # FEEL FREE TO CHANGE FORMATTING
    plt.savefig('ellipse.png' )  

