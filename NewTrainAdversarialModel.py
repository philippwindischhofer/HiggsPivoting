import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from base.Configs import TrainingConfig

def extract_shuffled_slice(sample, slice_def, random_state = 12345):
    shuffled_sample = sample.sample(frac = 1, random_state = random_state).reset_index(drop = True)
    cur_length = len(shuffled_sample)
    cur_slice = shuffled_sample[int(slice_def[0] * cur_length) : int(slice_def[1] * cur_length)]
    return cur_slice

def TrainAdversarialModel(infile_path, outdir, verbose_statistics = False):
    
    # prepare the training data
    tconf = TrainingConfig.from_file(outdir)
    data_branches = tconf.training_branches
    print("using data_branches = " + ", ".join(data_branches))

    # read the training data
    sig_sample_names = TrainingConfig.sig_samples
    bkg_sample_names = TrainingConfig.bkg_samples
    training_slice = TrainingConfig.training_slice
    validation_slice = TrainingConfig.validation_slice

    print("loading data ...")
    sig_data = [pd.read_hdf(infile_path, key = cur_sample) for cur_sample in sig_sample_names]
    bkg_data = [pd.read_hdf(infile_path, key = cur_sample) for cur_sample in bkg_sample_names]
    print("done!")

    # split it into training / validation slices
    sig_data_train = [extract_shuffled_slice(cur_sample, slice_def = training_slice) for cur_sample in sig_data]
    sig_data_val = [extract_shuffled_slice(cur_sample, slice_def = validation_slice) for cur_sample in sig_data]

    bkg_data_train = [extract_shuffled_slice(cur_sample, slice_def = training_slice) for cur_sample in bkg_data]
    bkg_data_val = [extract_shuffled_slice(cur_sample, slice_def = validation_slice) for cur_sample in bkg_data]

    # test a few things
    testpath = "/data/atlassmallfiles/users/windischhofer/HiggsPivotingPaper/Hbb_adv_madgraph_2020_04_04_new_commissioning"

    tconf = TrainingConfig.from_file(testpath)

    from models.ModelCollection import ModelCollection
    mcoll = ModelCollection.from_config(testpath)

    from training.ModelCollectionTrainer import ModelCollectionTrainer
    from training.BatchSamplers import sample_from_TrainingSamples
    trainer = ModelCollectionTrainer(mcoll, batch_sampler = sample_from_TrainingSamples, 
                                     training_pars = tconf.training_pars)
    trainer.train(sig_data_train, bkg_data_train, sig_data_val, bkg_data_val)

    # from models.AdversarialModel import AdversarialModel
    # mod = AdversarialModel.from_config(testpath)

    # from training.AdversarialModelTrainer import AdversarialModelTrainer

    # trainer = AdversarialModelTrainer(mod, batch_sampler = sample_from_TrainingSamples)
    # trainer.train(sig_data_train, bkg_data_train, sig_data_val, bkg_data_val)

    # from models.ModelCollection import ModelCollection
    # mcoll = ModelCollection([mod])

    # pred = mcoll.predict(sig_data_train[0][0:20])
    # print(pred)

    # # test a few things
    # testpath_2 = "/data/atlassmallfiles/users/windischhofer/HiggsPivotingPaper/Hbb_adv_madgraph_2020_04_04_new_commissioning"
    # from models.AdversarialModel import AdversarialModel
    # mod_2 = AdversarialModel.from_config(testpath_2)

    # trainer_2 = AdversarialModelTrainer(mod_2, batch_sampler = sample_from_TrainingSamples)
    # trainer_2.train(sig_data_train, bkg_data_train, sig_data_val, bkg_data_val)


if __name__ == "__main__":
    parser = ArgumentParser(description = "train adversarial networks")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--statistics", action = "store_const", const = True, default = False, dest = "verbose_statistics")
    args = vars(parser.parse_args())

    TrainAdversarialModel(**args)
