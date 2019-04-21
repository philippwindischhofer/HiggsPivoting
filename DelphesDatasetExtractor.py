import numpy as np
import uproot as ur
import pandas as pd
import re, os, glob

from argparse import ArgumentParser
from configparser import ConfigParser

from delphes.CrossSectionReader import CrossSectionReader
from delphes.Hbb0LepDelphesPreprocessor import Hbb0LepDelphesPreprocessor
from delphes.Hbb1LepDelphesPreprocessor import Hbb1LepDelphesPreprocessor

def PrepareDelphesDataset(input_files, lumifile_path, channel):
    """ Return a pandas table with the needed event variables, after applying selection. """

    if lumifile_path:
        print("using the following lumifile: '{}'".format(lumifile_path))

        # read in the lumi weight from the lumifile
        lumiconfig = ConfigParser()
        lumiconfig.read(lumifile_path)
    
        xsec = float(lumiconfig["global"]["xsec"])
        lumi = float(lumiconfig["global"]["lumi"]) # stored in fb^-1
        sow = float(lumiconfig["global"]["sow"]) # stored in pb

        # compute the lumiweight
        lumiweight = xsec * (lumi * 1000.0) / sow
    else:
        lumiweight = 1.0 # if no lumifile given, retain the original event weights

    print("using the following lumi event weight: {}".format(lumiweight))

    # look for the ROOT file(s) with the events and process it
    processed_events = []

    if channel == "0lep":
        pre = Hbb0LepDelphesPreprocessor()
    elif channel == "1lep":
        pre = Hbb1LepDelphesPreprocessor()
    else:
        raise Exception("The requested channel is not supported!")

    for event_file_candidate in input_files:
        print("currently processing {}".format(event_file_candidate))
        
        pre.load(event_file_candidate)
        processed = pre.process(lumiweight = lumiweight)
        if processed is not None:
            print("got {} processed events".format(len(processed)))
            processed_events.append(processed)

    # this will return a Pandas dataframe
    if len(processed_events) > 0:
        retval = pd.concat(processed_events).reset_index(drop = True)
    else:
        retval = None

    return retval

if __name__ == "__main__":
    parser = ArgumentParser(description = "convert Delphes datasets into hdf5, applying some event selection")
    parser.add_argument("--outfile", action = "store", dest = "outfile")
    parser.add_argument("--lumifile", action = "store", dest = "lumifile", default = None)
    parser.add_argument("--sname", action = "store", dest = "sample_name")
    parser.add_argument("--channel", action = "store", dest = "channel", default = "0lep")
    parser.add_argument("files", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    outfile_path = args["outfile"]
    lumifile_path = args["lumifile"]
    files = args["files"]
    sample_name = args["sample_name"]
    channel = args["channel"]

    processed_events = PrepareDelphesDataset(files, lumifile_path, channel)

    # merge all events together and dump them
    # Note: if no events survived the selection, NO output will be written!
    if processed_events is not None:
        print("finished processing, here is a sample:")
        print(processed_events.head())

        print("stored {} events".format(len(processed_events)))
        
        if os.path.exists(outfile_path):
            mode = 'a'
        else:
            mode = 'w'
            
        processed_events.to_hdf(outfile_path, key = sample_name, mode = mode)
