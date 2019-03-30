import numpy as np
import uproot as ur
import pandas as pd
import re, os, glob

from argparse import ArgumentParser
from configparser import ConfigParser

from delphes.CrossSectionReader import CrossSectionReader
from delphes.Hbb0LepDelphesPreprocessor import Hbb0LepDelphesPreprocessor

def PrepareDelphesDataset(input_dir, lumi):
    """ Return a pandas table with the needed event variables, after applying selection. """

    # look for the file containing the cross section
    xsec_file_candidates = glob.glob(os.path.join(input_dir, "cross_section.txt"))
    if len(xsec_file_candidates) != 1:
        raise Exception("Error: found more than one plausible cross-section file. Please check your files!")
    else:
        xsec_file_path = xsec_file_candidates[0]

    print("using the following cross-section file: '{}'".format(xsec_file_path))
    metadata = CrossSectionReader.parse(xsec_file_path)
    
    print("found the following metadata:")
    for key, val in metadata.items():
        print("{} = {}".format(key, val))

    xsec = float(metadata["cross"])

    # look for the ROOT file(s) with the events and process it
    processed_events = []

    event_file_candidates = glob.glob(os.path.join(input_dir, "**/*.root"), recursive = True)
    print("found {} ROOT files for this process".format(len(event_file_candidates)))

    pre = Hbb0LepDelphesPreprocessor()
    for event_file_candidate in event_file_candidates:
        print("currently processing {}".format(event_file_candidate))
        
        pre.load(event_file_candidate)
        processed_events.append(pre.process(lumi = lumi, xsec = xsec))

    # this will return a Pandas dataframe
    retval = pd.concat(processed_events).reset_index(drop = True)
    return retval

if __name__ == "__main__":
    parser = ArgumentParser(description = "convert Delphes datasets into hdf5, applying some event selection")
    parser.add_argument("--outfile", action = "store", dest = "outfile")
    parser.add_argument("--config", action = "store", dest = "config")
    parser.add_argument("--lumi", action = "store", dest = "lumi")
    args = vars(parser.parse_args())

    outfile_path = args["outfile"]
    config_file = args["config"]
    lumi = float(args["lumi"])
    
    # parse the configuration file
    config = ConfigParser()
    config.read(config_file)

    for name, section in config.items():

        # check if this section defines a sample
        sample_re = re.compile("sample:(.+)")
        m = sample_re.match(name)

        if not m:
            continue

        sample_name = m.group(1)

        print("----------------------------------------------------")
        print("found configuration for sample '{}'".format(sample_name))

        input_dirs = eval(section["input_dirs"])

        processed_events = []

        for input_dir in input_dirs:
            print("now processing input '{}'".format(input_dir))

            processed_events.append(PrepareDelphesDataset(input_dir, lumi))

        # merge all events together and dump them
        merged_events = pd.concat(processed_events).reset_index(drop = True)
        print("finished processing, here is a sample:")
        print(merged_events.head())
        
        if os.path.exists(outfile_path):
            mode = 'a'
        else:
            mode = 'w'

        merged_events.to_hdf(outfile_path, key = sample_name, mode = mode)

        print("----------------------------------------------------")

