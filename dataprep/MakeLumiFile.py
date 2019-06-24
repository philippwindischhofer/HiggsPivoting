import os, glob
from argparse import ArgumentParser
from configparser import ConfigParser

from delphes.CrossSectionReader import CrossSectionReader
from delphes.Hbb0LepDelphesPreprocessor import Hbb0LepDelphesPreprocessor

def GenerateLumiFile(input_dir, lumi, xsec = None):
    # if no explicit cross section given, try to read it from the generator output
    if not xsec:
        # first, read the generator-level cross section
        print("looking for xsec file in '{}'".format(input_dir))
        xsec_file_path = os.path.join(input_dir, "cross_section.txt")
        print("using the following cross-section file: '{}'".format(xsec_file_path))
        metadata = CrossSectionReader.parse(xsec_file_path)
    
        print("found the following metadata:")
        for key, val in metadata.items():
            print("{} = {}".format(key, val))

        try:
            xsec = float(metadata["cross"])
        except KeyError:
            print("Error: cross section information could not be found in generator output.")
    
    # then, compute the total generator-level SOW (= SOW at Delphes-level, since all events are propagated through the detector simulation):
    event_file_candidates = glob.glob(os.path.join(input_dir, "**/*.root"), recursive = True)
    print("found {} ROOT files for this process".format(len(event_file_candidates)))

    pre = Hbb0LepDelphesPreprocessor()
    
    SOW = 0.0
    for event_file in event_file_candidates:
        pre.load(event_file)
        cur_sow = pre.get_SOW()
        print("found SOW = {} in '{}'".format(cur_sow, event_file))

        SOW += cur_sow

    # finally, write the lumi file to disk
    lumiconfig = ConfigParser()
    lumiconfig["global"] = {"xsec": str(xsec), "lumi": str(lumi), "SOW": str(SOW)}

    lumiconfig_path = os.path.join(input_dir, "lumi.conf")
    with open(lumiconfig_path, "w") as outfile:
        lumiconfig.write(outfile)

if __name__ == "__main__":
    parser = ArgumentParser(description = "generate lumifile for an entire dataset")
    parser.add_argument("dirs", nargs = '+', action = "store")
    parser.add_argument("--lumi", action = "store", dest = "lumi")
    parser.add_argument("--xsec", action = "store", dest = "xsec")
    args = vars(parser.parse_args())

    dirs = args["dirs"]
    lumi = float(args["lumi"])
    xsec = args["xsec"]
    if xsec:
        xsec = float(xsec)

    print("generate lumi files for {} directories".format(len(dirs)))

    # generate a lumi file for each directory sequentially
    for cur_dir in dirs:
        GenerateLumiFile(cur_dir, lumi, xsec)
