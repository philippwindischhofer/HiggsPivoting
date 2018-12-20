import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from MINEClassifierEnvironment import MINEClassifierEnvironment
from SimpleModel import SimpleModel
from AdversarialTrainer import AdversarialTrainer

from Configs import TrainingConfig
        
def main():
    infile_path = "/data/atlas/atlasdata/windischhofer/Hbb/training-mc16d.h5"
    outdir = "/home/windischhofer/HiggsPivoting/adversarial_models"

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

    # set up the training environment
    mod = SimpleModel("test_model", hyperpars = {"num_hidden_layers": 3, "num_units": 30})
    mce = MINEClassifierEnvironment(classifier_model = mod)
    mce.build(num_inputs = len(data_branches), num_nuisances = 1)

    # set up the training
    train = AdversarialTrainer(training_pars = {"batch_size": 256})
    train.train(mce, number_batches = 100, df_sig = sig_data_train, df_bkg = bkg_data_train, nuisances = ["mBB"])

    mce.save(os.path.join(outdir, "test_model.dat"))

if __name__ == "__main__":
    main()
