import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from PCAWhiteningPreprocessor import PCAWhiteningPreprocessor
from SimpleClassifierEnvironment import SimpleClassifierEnvironment
from SimpleModel import SimpleModel
from Trainer import Trainer

from Configs import TrainingConfig
        
def main():
    infile_path = "/data/atlas/atlasdata/windischhofer/Hbb/training-mc16d.h5"
    outdir = "/home/windischhofer/HiggsPivoting/models"

    data_branches = TrainingConfig.training_branches

    # read the training data
    print("loading data ...")
    sig_data, bkg_data = pd.read_hdf(infile_path, key = 'sig'), pd.read_hdf(infile_path, key = 'bkg')
    print("got " + str(len(sig_data)) + " signal events")
    print("got " + str(len(bkg_data)) + " background events")

    # perform training / testing split
    test_size = 0.2
    sig_data_train, sig_data_test = train_test_split(sig_data, test_size = test_size)
    bkg_data_train, bkg_data_test = train_test_split(bkg_data, test_size = test_size)

    # set up the PCA whitening on on the training data
    pca_pre = PCAWhiteningPreprocessor(data_branches = data_branches)
    pca_pre.setup(np.concatenate([sig_data_train, bkg_data_train], axis = 0))
    pca_pre.save(os.path.join(outdir, "pre.pkl"))

    # apply the PCA to training and test data
    sig_data_train = pca_pre.process(sig_data_train)
    bkg_data_train = pca_pre.process(bkg_data_train)

    sig_data_test = pca_pre.process(sig_data_test)
    bkg_data_test = pca_pre.process(bkg_data_test)

    # set up the training environment
    mod = SimpleModel("test_model", hyperpars = {"num_hidden_layers": 3, "num_units": 30})
    sce = SimpleClassifierEnvironment(classifier_model = mod)
    sce.build(num_inputs = len(data_branches))

    # set up the training
    train = Trainer(training_pars = {"batch_size": 256})
    train.train(sce, number_batches = 10000, data_sig = sig_data_train, data_bkg = bkg_data_train)

    sce.save(os.path.join(outdir, "test_model.dat"))

if __name__ == "__main__":
    main()
