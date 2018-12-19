from sklearn.decomposition import PCA
import numpy as np
import pickle

from Preprocessor import Preprocessor

class PCAWhiteningPreprocessor(Preprocessor):

    def __init__(self, data_branches = None):
        if data_branches is not None:
            self.pca = PCA(n_components = len(data_branches), svd_solver = 'auto', whiten = True)
        else:
            self.pca = None

    @classmethod
    def from_file(cls, filepath):
        obj = cls()
        with open(filepath, "rb") as infile:
            obj.pca = pickle.load(infile)
        return obj

    def save(self, filepath):
        with open(filepath, "wb") as outfile:
            pickle.dump(self.pca, outfile)

    # set up the PCA on this data
    def setup(self, data):
        self.pca.fit(data)

    def process(self, chunk):
        processed_data = self.pca.transform(chunk)
        return processed_data

    
