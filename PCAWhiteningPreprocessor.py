from sklearn.decomposition import PCA
import numpy as np

from Preprocessor import Preprocessor

class PCAWhiteningPreprocessor(Preprocessor):

    def __init__(self, data_branches):
        self.pca = PCA(n_components = len(data_branches), svd_solver = 'auto', whiten = True)

    # set up the PCA on this data
    def setup(self, data):
        self.pca.fit(data)

    def process(self, chunk):
        processed_data = self.pca.transform(chunk)
        return processed_data
