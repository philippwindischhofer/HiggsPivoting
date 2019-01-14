import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from MINEClassifierEnvironment import MINEClassifierEnvironment

from SimpleModel import SimpleModel
from AdversarialTrainer import AdversarialTrainer

from ConfigFileUtils import ConfigFileUtils
from Configs import TrainingConfig
        
def main():
    parser = ArgumentParser(description = "train adversarial networks")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    args = vars(parser.parse_args())

    infile_path = args["infile_path"]
    outdir = args["outdir"]

    print("using infile_path = " + infile_path)
    print("using outdir = " + outdir)

    data_branches = TrainingConfig.training_branches
    print("using data_branches = " + ", ".join(data_branches))

    # read the training data
    print("loading data ...")
    sig_data, bkg_data = pd.read_hdf(infile_path, key = 'sig'), pd.read_hdf(infile_path, key = 'bkg')
    print("got " + str(len(sig_data)) + " signal events")
    print("got " + str(len(bkg_data)) + " background events")

    # perform training / testing split
    test_size = 0.2
    sig_data_train, sig_data_test = train_test_split(sig_data, test_size = test_size, random_state = 12345)
    bkg_data_train, bkg_data_test = train_test_split(bkg_data, test_size = test_size, random_state = 12345)

    # set up the training environment
    model_type = ConfigFileUtils.get_env_type(outdir)
    if model_type is not None:
        mce = model_type.from_file(outdir)
    else:
        print("no model type prescribed in this config file, using MINEClassifierEnvironment as default")
        mce = MINEClassifierEnvironment.from_file(outdir)

    # set up the training
    train = AdversarialTrainer(training_pars = {"batch_size": 1024, "pretrain_batches": 5000, "printout_interval": 1})
    train.train(mce, number_batches = 10000, df_sig = sig_data_train, df_bkg = bkg_data_train, nuisances = ["mBB"])

    # save all the necessary information
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    mce.save(os.path.join(outdir, ))
    train.save_training_statistics(os.path.join(outdir, "training_evolution.pkl"))

if __name__ == "__main__":
    main()
