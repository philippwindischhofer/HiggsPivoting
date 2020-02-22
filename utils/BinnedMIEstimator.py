import numpy as np
from scipy import ndimage
from scipy.stats import iqr
from functools import partial
from plotting.ModelEvaluator import ModelEvaluator

class BinnedMIEstimator:

    def __init__(self, name):
        self.name = name

    def _get_cellucci_binning(self, data):
        data = data.flatten()
        data_min = np.min(data)
        data_max = np.max(data)

        number_bins = int(np.sqrt(float(len(data)) / 5.0))
        percentile_getter = partial(ModelEvaluator._weighted_percentile, data, weights = np.ones_like(data))
         
        # test the binning with the current number of bins
        percentiles = np.linspace(0, 1, number_bins + 1)
        uniform_occupancy_binning = list(map(percentile_getter, percentiles))
        
        return uniform_occupancy_binning

    def estimate(self, X_in, Y_in, weights, bins_heuristic = ""):
        
        assert len(X_in) == len(Y_in)

        if bins_heuristic == "tukey":
            bins_X = bins_Y = int(np.sqrt(len(X_in)))
        elif bins_heuristic == "bendat_piersol":
            bins_X = bins_Y = int(1.87 * np.power(len(X_in) - 1, 0.4))
        elif bins_heuristic == "cellucci_approximated":
            bins_X = bins_Y = int(np.sqrt(float(len(X_in)) / 5.0))
        elif bins_heuristic == "cellucci":
            bins_X = self._get_cellucci_binning(X_in)
            bins_Y = self._get_cellucci_binning(Y_in)
        else:
            raise NotImplementedError("Error: selected heuristic not implemented!")

        bins = [bins_X, bins_Y]
        eps = 1e-6

        X_in = X_in.flatten()
        Y_in = Y_in.flatten()

        # estimate the densities with histograms
        p_XY = np.histogram2d(X_in, Y_in, bins = bins)[0].astype(float) + eps
        p_XY /= float(np.sum(p_XY))

        # get the marginals
        p_X = np.sum(p_XY, axis = 0).flatten().astype(float)
        p_X /= float(np.sum(p_X)) # note: will already be normalised correctly anyways!

        p_Y = np.sum(p_XY, axis = 1).flatten().astype(float)
        p_Y /= float(np.sum(p_Y)) # note: will already be normalised correctly anyways!

        MI = np.sum(p_XY * np.log(p_XY)) - np.sum(p_X * np.log(p_X)) - np.sum(p_Y * np.log(p_Y))

        return MI
