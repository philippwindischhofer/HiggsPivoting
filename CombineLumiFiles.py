import glob, os
from argparse import ArgumentParser
from configparser import ConfigParser

def CombineLumiFiles(indir):
    """ combines all lumi.conf files found in all subdirectories """

    # first, look for all existing lumi files
    lumifiles = glob.glob(os.path.join(indir, "**/lumi.conf"), recursive = True)
    print("have found {} lumi files in this directory".format(len(lumifiles)))

    # now, combine all lumi files: just add their SOW together
    # and make sure that the process cross sections and luminosities
    # agree between files
    SOW_combined = 0
    xsec_combined = -1
    lumi_combined = -1

    for cur_lumifile in lumifiles:
        cur_lumiconfig = ConfigParser()
        cur_lumiconfig.read(cur_lumifile)

        cur_SOW = cur_lumiconfig["global"]["SOW"]
        cur_xsec = cur_lumiconfig["global"]["xsec"]
        cur_lumi = cur_lumiconfig["global"]["lumi"]

        if xsec_combined < 0:
            xsec_combined = cur_xsec

        if lumi_combined < 0:
            lumi_combined = cur_lumi

        if xsec_combined != cur_xsec or lumi_combined != cur_lumi:
            raise Exception("Error: some of your lumi files are not compatible!")

        SOW_combined += cur_SOW

    lumiconfig_combined = ConfigParser()
    lumiconfig_combined["global"] = {"xsec": str(xsec_combined), "lumi": str(lumi_combined), "SOW": str(SOW_combined)}
    with open(os.path.join(indir, "lumi.conf"), "w") as combined_outfile:
        lumiconfig_combined.write(combined_outfile)

if __name__ == "__main__":
    parser = ArgumentParser(description = "generate lumifile on the batch system")
    parser.add_argument("dir", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    indir = args["dir"]
    assert len(indir) == 1 # works only with a single directory at a time
    indir = indir[0]

    CombineLumiFiles(indir)
