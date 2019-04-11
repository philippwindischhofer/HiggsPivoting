from argparse import ArgumentParser
from configparser import ConfigParser
import pandas as pd

def ApplyLumiWeight(lumifile, infile, outfile, name_in, name_out):
    # first, read in the lumifile and compute the event weight
    lumiconfig = ConfigParser()
    lumiconfig.read(lumifile)

    lumiweight = lumiconfig["xsec"] * lumiconfig["lumi"] / lumiconfig["SOW"]

    data = pd.read_hdf(infile, key = name_in)
    data["EventWeight"] *= lumiweight
    data.to_hdf(outfile, key = name_out, mode = 'w')

if __name__ == "__main__":
    parser = ArgumentParser(description = "weight events according to the total luminosity")
    parser.add_argument("--lumifile", action = "store", dest = "lumifile")
    parser.add_argument("--infile", action = "store", dest = "infile")
    parser.add_argument("--outfile", action = "store", dest = "outfile")
    parser.add_argument("--name_in", action = "store", dest = "name_in")
    parser.add_argument("--name_out", action = "store", dest = "name_out")
    args = vars(parser.parse_args())

    ApplyLumiWeight(**args)
