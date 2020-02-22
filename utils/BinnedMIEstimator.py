import numpy as np
from scipy import ndimage

class BinnedMIEstimator:

    def __init__(self, name):
        self.name = name

    def estimate(self, X_in, Y_in, weights):
        bins = (256, 256)
        eps = 1e-6

        X_in = X_in.flatten()
        Y_in = Y_in.flatten()

        # estimate the densities with histograms
        p_XY = np.histogram2d(X_in, Y_in, bins = bins)[0].astype(float) + eps
        p_XY /= float(np.sum(p_XY))

        p_X = np.histogram(X_in, bins = bins[0])[0].astype(float) + eps
        p_X /= float(np.sum(p_X))

        p_Y = np.histogram(Y_in, bins = bins[0])[0].astype(float) + eps
        p_Y /= float(np.sum(p_Y))

        MI = np.sum(p_XY * np.log(p_XY)) - np.sum(p_X * np.log(p_X)) - np.sum(p_Y * np.log(p_Y))

        return MI
