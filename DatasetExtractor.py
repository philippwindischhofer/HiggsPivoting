import numpy as np
import uproot as ur
import pandas as pd
from sklearn.model_selection import train_test_split

import base.Generators as gens
from base.SimplePreprocessor import SimplePreprocessor
from base.Configs import TrainingConfig

def TrainNuisAuxSplit(indat):
    return indat[TrainingConfig.training_branches].values, indat[TrainingConfig.nuisance_branches].values, indat[["EventWeight"]].values

def PrepareDataset():
    infile_path = "/home/windischhofer/datasmall/Hbb/hist-all-mc16a.root"
    outfile_path = "/home/windischhofer/datasmall/Hbb/training-mc16a.h5"

    # these are the branches needed for the training itself later on
    data_branches = TrainingConfig.training_branches + TrainingConfig.auxiliary_branches + TrainingConfig.other_branches
    truth_branches = ["Sample"]
    other_branches = []
    read_branches = data_branches + other_branches + truth_branches

    sample_defs = {"Hbb": ["ggZvvH125", "qqZvvH125", "qqWlvH125", "qqZllH125", "qqWlvH125cc", "ggZvvH125cc", "ggZllH125"], 
                   "Zjets": ["Zbb", "Zbc", "Zbl", "Zcc", "Zcl", "Zl"],
                   "Wjets": ["Wbb", "Wbc", "Wbl", "Wcc", "Wcl", "Wl"],
                   "ttbar": ["ttbar"],
                   "diboson": ["WW", "ZZ", "WZ", "ggWW", "ggZZ"],
                   "singletop": ["stopWt", "stopt", "stops"]}

    # prepare the necessary cuts for the individual samples
    cuts = []
    for ind, (sample_name, sample_def) in enumerate(sample_defs.items()):
        sample_def_bin = [str.encode(samp) for samp in sample_def]
        # note: need to pass 'sample_def_bin' explicitly as default argument, since otherwise, it would be referenced when the
        # lambda is called, not when it is created!
        cuts.append(lambda row, sample_def_bin = sample_def_bin: any([row["Sample"] == name for name in sample_def_bin]))

    # get the data and split it into signal and background
    print("reading ROOT tree")
    gen = gens.raw_data(infile_path, "Nominal", read_branches)
    pre = SimplePreprocessor(data_branches, cuts = cuts)

    cut_samples = pre.process_generator(gen, rettype = 'pd')

    for cut_sample, sample_name in zip(cut_samples, sample_defs.keys()):
        print("read {} events for sample '{}'".format(str(len(cut_sample)), sample_name))

    print("saving to hdf file ...")
    for ind, (cut_sample, sample_name) in enumerate(zip(cut_samples, sample_defs.keys())):
        cut_sample.to_hdf(outfile_path, key = sample_name, mode = 'w' if ind == 0 else 'a')
    print("done!")
        
if __name__ == "__main__":
    PrepareDataset()
