import numpy as np
import uproot as ur
import pandas as pd
from sklearn.model_selection import train_test_split

import Generators as gens
from SimplePreprocessor import SimplePreprocessor
from PCAWhiteningPreprocessor import PCAWhiteningPreprocessor
from TFEnvironment import TFEnvironment
from SimpleModel import SimpleModel
        
def main():
    file_path = "/data/atlas/atlasdata/windischhofer/Hbb/hist-all-mc16d.root"

    data_branches = ["mBB", "mB1", "mB2"]
    truth_branches = ["Sample"]
    read_branches = data_branches + truth_branches

    # sig_samples = ["qqZvvH125, qqWlvH125"]
    # bkg_samples = ["Zbb"]

    # for testing purposes
    sig_samples = ["Wl"]
    bkg_samples = ["Wbl"]

    # convert them into binary representations, to match the way they are read from the tree
    sig_samples_bin = [str.encode(samp) for samp in sig_samples]
    bkg_samples_bin = [str.encode(samp) for samp in bkg_samples]

    sig_cut = lambda row: any([name == row["Sample"] for name in sig_samples_bin])
    bkg_cut = lambda row: any([name == row["Sample"] for name in bkg_samples_bin])

    # get the data and split it into signal and background
    gen = gens.raw_data(file_path, "Nominal", read_branches)
    pre = SimplePreprocessor(data_branches = data_branches, sig_cut = sig_cut, bkg_cut = bkg_cut)
    sig_data, bkg_data = pre.process_generator(gen)

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
    tfe = TFEnvironment(classifier_model = mod, training_pars = {"batch_size": 256}, num_inputs = len(data_branches))

    tfe.build()
    #tfe.train(number_epochs = 300, data_sig = sig_data_train, data_bkg = bkg_data_train)
    #tfe.save("/home/windischhofer/HiggsPivoting/models/test_model.dat")
    tfe.load("/home/windischhofer/HiggsPivoting/models/test_model.dat")

    # now apply it on the test dataset
    pred = tfe.predict(data_test = bkg_data_test)
    print(pred)

    pred = tfe.predict(data_test = sig_data_test)
    print(pred)

if __name__ == "__main__":
    main()
