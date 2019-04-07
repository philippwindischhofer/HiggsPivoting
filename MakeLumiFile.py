import os, glob
from argparse import ArgumentParser
from configparser import ConfigParser

from delphes.CrossSectionReader import CrossSectionReader
from delphes.Hbb0LepDelphesPreprocessor import Hbb0LepDelphesPreprocessor

def GenerateLumiFile(input_dir, lumi):
    # first, read the generator-level cross section
    print("looking for xsec file in '{}'".format(input_dir))
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

    # this is the lumi * xsec event weight (lumi is expected to be provided in units of fb^-1, while the cross section from MadGraph is given in pb)
    evweight = lumi * 1000 * xsec / SOW

    # finally, write the lumi file to disk
    lumiconfig = ConfigParser()
    lumiconfig["global"] = {"xsec": str(xsec), "lumi": str(lumi), "SOW": str(SOW), "evweight": str(evweight)}

    lumiconfig_path = os.path.join(indir, "lumi.conf")
    with open(lumiconfig_path, "w") as outfile:
        lumiconfig.write(outfile)

if __name__ == "__main__":
    parser = ArgumentParser(description = "generate lumifile for an entire dataset")
    parser.add_argument("dirs", nargs = '+', action = "store")
    parser.add_argument("--lumi", action = "store", dest = "lumi")
    args = vars(parser.parse_args())

    dirs = args["dirs"]
    lumi = args["lumi"]

    # generate a lumi file for each directory sequentially
    for cur_dir in dirs:
        GenerateLumiFile(cur_dir, lumi)
