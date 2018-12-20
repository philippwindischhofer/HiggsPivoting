from sklearn.decomposition import PCA
import numpy as np
import pickle

from Preprocessor import Preprocessor

class PCAWhiteningPreprocessor(Preprocessor):

    def __init__(self, num_inputs = None):
        if num_inputs is not None:
            self.pca = PCA(n_components = num_inputs, svd_solver = 'auto', whiten = True)
        else:
            self.pca = None

    @classmethod
    def from_file(cls, filepath):
        print("attempting to read PCAWhiteningPreprocessor from " + filepath)
        obj = cls()
        with open(filepath, "rb") as infile:
            obj.pca = pickle.load(infile)
        return obj

    def save(self, filepath):
        print("writing PCAWhiteningPreprocessor to " + filepath)
        with open(filepath, "wb") as outfile:
            pickle.dump(self.pca, outfile)

    # set up the PCA on this data
    def setup(self, data):
        self.pca.fit(data)

    def process(self, chunk):
        processed_data = self.pca.transform(chunk)
        return processed_data

    
