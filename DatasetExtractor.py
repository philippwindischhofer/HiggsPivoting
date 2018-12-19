import numpy as np
import uproot as ur
import pandas as pd
from sklearn.model_selection import train_test_split

import Generators as gens
from SimplePreprocessor import SimplePreprocessor
from Configs import TrainingConfig
        
def main():
    infile_path = "/data/atlas/atlasdata/windischhofer/Hbb/hist-all-mc16d.root"
    outfile_path = "/data/atlas/atlasdata/windischhofer/Hbb/training-mc16d.h5"

    data_branches = TrainingConfig.training_branches
    truth_branches = ["Sample"]
    read_branches = data_branches + truth_branches

    # for testing purposes
    sig_samples = ["ggZvvH125", "qqZvvH125"]
    bkg_samples = ["Zbb"]

    # convert them into binary representations, to match the way they are read from the tree
    sig_samples_bin = [str.encode(samp) for samp in sig_samples]
    bkg_samples_bin = [str.encode(samp) for samp in bkg_samples]

    sig_cut = lambda row: any([name == row["Sample"] for name in sig_samples_bin])
    bkg_cut = lambda row: any([name == row["Sample"] for name in bkg_samples_bin])

    # get the data and split it into signal and background
    print("reading ROOT tree")
    gen = gens.raw_data(infile_path, "Nominal", read_branches)
    pre = SimplePreprocessor(data_branches, sig_cut, bkg_cut)

    sig_data, bkg_data = pre.process_generator(gen, rettype = 'pd')

    print("read " + str(len(sig_data)) + " signal events")
    print("read " + str(len(bkg_data)) + " background events")

    print("saving to hdf file ...")
    sig_data.to_hdf(outfile_path, key = 'sig', mode = 'w')
    bkg_data.to_hdf(outfile_path, key = 'bkg', mode = 'a')
    print("done!")
        
if __name__ == "__main__":
    main()
