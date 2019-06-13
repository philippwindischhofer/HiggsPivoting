import os
import numpy as np
import pandas as pd
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
    sig_samples = TrainingConfig.sig_samples
    bkg_samples = TrainingConfig.bkg_samples
    training_slice = TrainingConfig.training_slice

    print("loading data ...")
    sig_data = [pd.read_hdf(infile_path, key = sig_sample) for sig_sample in sig_samples]
    bkg_data = [pd.read_hdf(infile_path, key = bkg_sample) for bkg_sample in bkg_samples]

    auxdat_sig = []
    auxdat_bkg = []

    # extract the training dataset
    sig_data_train = []
    for sample, sample_name in zip(sig_data, sig_samples):
        cur_length = len(sample)
        sample = sample.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample
        cur_train = sample[int(training_slice[0] * cur_length) : int(training_slice[1] * cur_length)]
        auxdat_sig.append(cur_train[TrainingConfig.auxiliary_branches].values)
        sig_data_train.append(cur_train)

    bkg_data_train = []
    for sample, sample_name in zip(bkg_data, bkg_samples):
        cur_length = len(sample)
        sample = sample.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample
        cur_train = sample[int(training_slice[0] * cur_length) : int(training_slice[1] * cur_length)]
        auxdat_bkg.append(cur_train[TrainingConfig.auxiliary_branches].values)
        bkg_data_train.append(cur_train)

    print("got " + str(len(sig_data)) + " signal datasets")
    print("got " + str(len(bkg_data)) + " background datasets")

    # split the dataset into training branches, nuisances and event weights
    traindat_sig = []
    nuisdat_sig = []
    weightdat_sig = []

    traindat_bkg = []
    nuisdat_bkg = []
    weightdat_bkg = []

    for cur_sig_data_train, sample_name in zip(sig_data_train, sig_samples):
        cur_traindat_sig, cur_nuisdat_sig, cur_weightdat_sig = TrainNuisAuxSplit(cur_sig_data_train)
        traindat_sig.append(cur_traindat_sig)
        nuisdat_sig.append(cur_nuisdat_sig)
        weightdat_sig.append(cur_weightdat_sig * TrainingConfig.sample_reweighting[sample_name])
        print("'{}' with {} entries representing {} events".format(sample_name, len(cur_weightdat_sig), np.sum(cur_weightdat_sig)))

    for cur_bkg_data_train, sample_name in zip(bkg_data_train, bkg_samples):
        cur_traindat_bkg, cur_nuisdat_bkg, cur_weightdat_bkg = TrainNuisAuxSplit(cur_bkg_data_train)
        traindat_bkg.append(cur_traindat_bkg)
        nuisdat_bkg.append(cur_nuisdat_bkg)
        weightdat_bkg.append(cur_weightdat_bkg * TrainingConfig.sample_reweighting[sample_name])
        print("'{}' with {} entries representing {} events".format(sample_name, len(cur_weightdat_bkg), np.sum(cur_weightdat_bkg)))

    print("starting up")
    mce = AdversarialEnvironment.from_file(outdir)

    training_pars = tconf.training_pars
    print("using the following training parameters:")
    for key, val in training_pars.items():
        print(key + " = " + str(val))

    # set up the training
    train = AdversarialTrainer(training_pars = training_pars)

    # give the full list of signal / background components to the trainer
    train.train(mce, number_batches = training_pars["training_batches"], traindat_sig = traindat_sig, traindat_bkg = traindat_bkg, 
                nuisances_sig = nuisdat_sig, nuisances_bkg = nuisdat_bkg, weights_sig = weightdat_sig, weights_bkg = weightdat_bkg, auxdat_sig = auxdat_sig, auxdat_bkg = auxdat_bkg,
                sig_sampling_pars = {"sampling_lengths": TrainingConfig.sig_sampling_lengths}, bkg_sampling_pars = {"sampling_lengths": TrainingConfig.bkg_sampling_lengths})

    # save all the necessary information
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    mce.save(os.path.join(outdir, ))
    train.save_training_statistics(os.path.join(outdir, "training_evolution.pkl"))

if __name__ == "__main__":
    main()
