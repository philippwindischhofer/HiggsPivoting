import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from base.Configs import TrainingConfig
from DatasetExtractor import TrainNuisAuxSplit

def ShowEventContent(infile_path, name):
    with pd.HDFStore(infile_path) as hdf:
        keys = hdf.keys()
        available_tables = [os.path.basename(key) for key in keys]

    for name in available_tables:
        data = pd.read_hdf(infile_path, key = name)

        testdata, nuisdata, weights = TrainNuisAuxSplit(data)
        cur_aux_data = data[TrainingConfig.other_branches].values
        
        total_events = np.sum(weights)
        print("{}: total events = {}".format(name, total_events))

if __name__ == "__main__":
    parser = ArgumentParser(description = "show events contained in this file")
    parser.add_argument("--infile", action = "store", dest = "infile_path")
    parser.add_argument("--name", action = "store", dest = "name", default = "generic_process")
    args = vars(parser.parse_args())

    ShowEventContent(**args)
