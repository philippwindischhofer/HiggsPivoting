import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from base.Configs import TrainingConfig
from DatasetExtractor import TrainNuisAuxSplit

def ShowEventContent(infile_path):
    with pd.HDFStore(infile_path) as hdf:
        keys = hdf.keys()
        available_tables = [os.path.basename(key) for key in keys]

    for name in available_tables:
        data = pd.read_hdf(infile_path, key = name)

        testdata, nuisdata, weights = TrainNuisAuxSplit(data)
        
        total_events = np.sum(weights)
        unweighted_events = len(weights)
        print("{}: total events = {} ({} unweighted)".format(name, total_events, unweighted_events))

if __name__ == "__main__":
    parser = ArgumentParser(description = "show events contained in this file")
    parser.add_argument("--infile", action = "store", dest = "infile_path")

    args = vars(parser.parse_args())

    ShowEventContent(**args)
