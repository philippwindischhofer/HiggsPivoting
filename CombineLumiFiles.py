import glob, os
from argparse import ArgumentParser
from configparser import ConfigParser

def IsGoodLumiFile(lumifile):
    try:
        lumiconfig = ConfigParser()
        lumiconfig.read(lumifile)
    
        lumisection = lumiconfig["global"]
        if "SOW" in lumisection and "xsec" in lumisection and "lumi" in lumisection:
            return True
        else:
            return False
    except (IOError, KeyError):
        return False

def CombineLumiFiles(indir, channel):
    from CombineEventFiles import IsGoodEventFile

    """ combines all lumi.conf files found in all subdirectories """

    # the assumed event file names
    eventfile = {"0lep": "events_0lep.h5",
                 "1lep": "events_1lep.h5"}

    # first, look for all existing lumi files
    # Note: this semi-automatic way of doing it is faster than simply
    #     lumifiles = glob.glob(os.path.join(indir, "**/lumi.conf"), recursive = True)
    sub_dirs = glob.glob(os.path.join(indir, '*/'))
    lumifiles = []
    for sub_dir in sub_dirs:
        lumifile_path = os.path.join(sub_dir, "lumi.conf")
        eventfile_path = os.path.join(sub_dir, eventfile[channel])
        if IsGoodLumiFile(lumifile_path) and IsGoodEventFile(eventfile_path):
            lumifiles.append(lumifile_path)

    print("have found {} lumi files in this directory".format(len(lumifiles)))

    # now, combine all lumi files: just add their SOW together
    # and make sure that the process cross sections and luminosities
    # agree between files
    SOW_combined = 0
    xsec_combined = -1
    lumi_combined = -1

    for cur_lumifile in lumifiles:
        if IsGoodLumiFile(cur_lumifile):
            cur_lumiconfig = ConfigParser()
            cur_lumiconfig.read(cur_lumifile)

            cur_SOW = float(cur_lumiconfig["global"]["SOW"])
            cur_xsec = float(cur_lumiconfig["global"]["xsec"])
            cur_lumi = float(cur_lumiconfig["global"]["lumi"])

            if xsec_combined < 0:
                xsec_combined = cur_xsec

            if lumi_combined < 0:
                lumi_combined = cur_lumi

            if xsec_combined != cur_xsec or lumi_combined != cur_lumi:
                raise Exception("Error: some of your lumi files are not compatible!")

            SOW_combined += cur_SOW
        else:
            print("'{}' is not a good lumifile, ignore it".format(cur_lumifile))

    lumiconfig_combined = ConfigParser()
    lumiconfig_combined["global"] = {"xsec": str(xsec_combined), "lumi": str(lumi_combined), "SOW": str(SOW_combined)}
    with open(os.path.join(indir, "lumi.conf"), "w") as combined_outfile:
        lumiconfig_combined.write(combined_outfile)

if __name__ == "__main__":
    parser = ArgumentParser(description = "combine lumi files")
    parser.add_argument("--channel", action = "store", default = "0lep")
    parser.add_argument("indir", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    indir = args["indir"]
    assert len(indir) == 1 # works only with a single directory at a time
    args["indir"] = indir[0]

    CombineLumiFiles(**args)
