import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from models.AdversarialEnvironment import AdversarialEnvironment
from training.AdversarialTrainer import AdversarialTrainer
from base.Configs import TrainingConfig
from DatasetExtractor import TrainNuisAuxSplit
        
def main():
    parser = ArgumentParser(description = "train adversarial networks")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    args = vars(parser.parse_args())

    infile_path = args["infile_path"]
    outdir = args["outdir"]

    print("using infile_path = " + infile_path)
    print("using outdir = " + outdir)

    tconf = TrainingConfig.from_file(outdir)
    data_branches = tconf.training_branches
    print("using data_branches = " + ", ".join(data_branches))

    # read the training data
    sig_samples = ["Hbb"]
    bkg_samples = ["ttbar", "Zjets", "Wjets", "diboson", "singletop"]

    print("loading data ...")
    sig_data = [pd.read_hdf(infile_path, key = sig_sample) for sig_sample in sig_samples]
    bkg_data = [pd.read_hdf(infile_path, key = bkg_sample) for bkg_sample in bkg_samples]

    # extract the training dataset
    test_size = TrainingConfig.test_size
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

    # split the dataset into training branches, nuisances and event weights
    traindat_sig, nuisdat_sig, weightdat_sig = TrainNuisAuxSplit(sig_data_train)
    traindat_bkg, nuisdat_bkg, weightdat_bkg = TrainNuisAuxSplit(bkg_data_train)

    print("starting up")
    mce = AdversarialEnvironment.from_file(outdir)

    training_pars = tconf.training_pars
    print("using the following training parameters:")
    for key, val in training_pars.items():
        print(key + " = " + str(val))

    # set up the training
    train = AdversarialTrainer(training_pars = training_pars)
    train.train(mce, number_batches = training_pars["training_batches"], traindat_sig = traindat_sig, traindat_bkg = traindat_bkg, nuisances_sig = nuisdat_sig, nuisances_bkg = nuisdat_bkg, weights_sig = weightdat_sig, weights_bkg = weightdat_bkg)

    # save all the necessary information
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    mce.save(os.path.join(outdir, ))
    train.save_training_statistics(os.path.join(outdir, "training_evolution.pkl"))

if __name__ == "__main__":
    main()
