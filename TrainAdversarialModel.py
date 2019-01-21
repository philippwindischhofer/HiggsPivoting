import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from models.AdversarialEnvironment import AdversarialEnvironment
from training.AdversarialTrainer import AdversarialTrainer
from base.Configs import TrainingConfig
        
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
    sig_samples = ["Hbb"]
    bkg_samples = ["ttbar", "Zjets", "Wjets", "diboson", "singletop"]

    print("loading data ...")
    sig_data = [pd.read_hdf(infile_path, key = sig_sample) for sig_sample in sig_samples]
    bkg_data = [pd.read_hdf(infile_path, key = bkg_sample) for bkg_sample in bkg_samples]

    # extract the training dataset
    test_size = 0.2
    sig_data_train = []
    for sample in sig_data:
        cur_train, _ = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        sig_data_train.append(cur_train)

    bkg_data_train = []
    for sample in bkg_data:
        cur_train, _ = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        bkg_data_train.append(cur_train)

    sig_data_train = pd.concat(sig_data_train)
    bkg_data_train = pd.concat(bkg_data_train)

    # shuffle them separately (particularly important for more than one background component)
    sig_data_train = sig_data_train.sample(frac = 1, random_state = 12345).reset_index(drop = True)
    bkg_data_train = bkg_data_train.sample(frac = 1, random_state = 12345).reset_index(drop = True)

    print("got " + str(len(sig_data)) + " signal datasets")
    print("got " + str(len(bkg_data)) + " background datasets")

    print("starting up")
    mce = AdversarialEnvironment.from_file(outdir)

    # set up the training
    train = AdversarialTrainer(training_pars = {"batch_size": 1024, "pretrain_batches": 300, "printout_interval": 100})
    train.train(mce, number_batches = 2000, df_sig = sig_data_train, df_bkg = bkg_data_train, nuisances = ["mBB"])

    # save all the necessary information
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    mce.save(os.path.join(outdir, ))
    train.save_training_statistics(os.path.join(outdir, "training_evolution.pkl"))

if __name__ == "__main__":
    main()
