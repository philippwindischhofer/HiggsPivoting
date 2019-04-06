import numpy as np
import uproot as ur
import pandas as pd
import re, os, glob

from argparse import ArgumentParser
from configparser import ConfigParser

from delphes.CrossSectionReader import CrossSectionReader
from delphes.Hbb0LepDelphesPreprocessor import Hbb0LepDelphesPreprocessor

def PrepareDelphesDataset(xsec_file_path, input_files, lumi):
    """ Return a pandas table with the needed event variables, after applying selection. """

    print("using the following cross-section file: '{}'".format(xsec_file_path))
    metadata = CrossSectionReader.parse(xsec_file_path)
    
    print("found the following metadata:")
    for key, val in metadata.items():
        print("{} = {}".format(key, val))

    xsec = float(metadata["cross"])

    # look for the ROOT file(s) with the events and process it
    processed_events = []

    pre = Hbb0LepDelphesPreprocessor()
    for event_file_candidate in input_files:
        print("currently processing {}".format(event_file_candidate))
        
        pre.load(event_file_candidate)
        processed = pre.process(lumi = lumi, xsec = xsec)
        print("got {} processed events".format(len(processed)))
        processed_events.append(processed)

    # this will return a Pandas dataframe
    retval = pd.concat(processed_events).reset_index(drop = True)
    return retval

if __name__ == "__main__":
    parser = ArgumentParser(description = "convert Delphes datasets into hdf5, applying some event selection")
    parser.add_argument("--outfile", action = "store", dest = "outfile")
    parser.add_argument("--xsec", action = "store", dest = "xsec_file_path")
    parser.add_argument("--lumi", action = "store", dest = "lumi")
    parser.add_argument("--sname", action = "store", dest = "sample_name")
    parser.add_argument("files", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    outfile_path = args["outfile"]
    lumi = float(args["lumi"])
    xsec_file_path = args["xsec_file_path"]
    files = args["files"]
    sample_name = args["sample_name"]
    
    processed_events = []

    processed_events.append(PrepareDelphesDataset(xsec_file_path, files, lumi))

    # merge all events together and dump them
    merged_events = pd.concat(processed_events).reset_index(drop = True)
    print("finished processing, here is a sample:")
    print(merged_events.head())

    print("stored {} events".format(len(merged_events)))
        
    if os.path.exists(outfile_path):
        mode = 'a'
    else:
        mode = 'w'
            
    merged_events.to_hdf(outfile_path, key = sample_name, mode = mode)
