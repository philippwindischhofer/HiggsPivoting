import numpy as np
import uproot as ur
import pandas as pd
from sklearn.model_selection import train_test_split

import Generators as gens
from SimplePreprocessor import SimplePreprocessor
from PCAWhiteningPreprocessor import PCAWhiteningPreprocessor
from SimpleClassifierEnvironment import SimpleClassifierEnvironment
from SimpleModel import SimpleModel
        
def main():
    infile_path = "/data/atlas/atlasdata/windischhofer/Hbb/training-mc16d.h5"

    data_branches = ["mBB", "mB1", "mB2"]

    # read the training data
    sig_data, bkg_data = pd.read_hdf(infile_path, key = 'sig'), pd.read_hdf(infile_path, key = 'bkg')

    print(sig_data)
    print(bkg_data)

    # perform training / testing split
    test_size = 0.2
    sig_data_train, sig_data_test = train_test_split(sig_data, test_size = test_size)
    bkg_data_train, bkg_data_test = train_test_split(bkg_data, test_size = test_size)

    # set up the PCA whitening on on the training data
    pca_pre = PCAWhiteningPreprocessor(data_branches = data_branches)
    pca_pre.setup(np.concatenate([sig_data_train, bkg_data_train], axis = 0))

    # apply the PCA again
    sig_data_train = pca_pre.process(sig_data_train)
    bkg_data_train = pca_pre.process(bkg_data_train)

    # set up the training environment
    mod = SimpleModel("test_model", hyperpars = {"num_hidden_layers": 3, "num_units": 30})
    sce = SimpleClassifierEnvironment(classifier_model = mod, training_pars = {"batch_size": 256}, num_inputs = len(data_branches))

    sce.build()
    sce.train(number_epochs = 300, data_sig = sig_data_train, data_bkg = bkg_data_train)
    sce.save("/home/windischhofer/HiggsPivoting/models/test_model.dat")
    sce.load("/home/windischhofer/HiggsPivoting/models/test_model.dat")

    # now apply it on the test dataset
    pred = sce.predict(data_test = bkg_data_test)
    print(pred)

    pred = sce.predict(data_test = sig_data_test)
    print(pred)

if __name__ == "__main__":
    main()
