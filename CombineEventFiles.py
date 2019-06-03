import os, glob
from argparse import ArgumentParser

from h5add import h5add

def IsGoodEventFile(eventfile):
    try:
        import pandas as pd
        df = pd.read_hdf(eventfile)
        return True
    except:
        return False

def CombineEventFiles(indir, channel):
    from CombineLumiFiles import IsGoodLumiFile

    # the assumed event file names
    eventfile = {"0lep": "events_0lep.h5",
                 "1lep": "events_1lep.h5"}

    assert len(indir) == 1
    indir = indir[0]

    # first, look for all existing lumi files
    # Note: this semi-automatic way of doing it is faster than simply
    #     lumifiles = glob.glob(os.path.join(indir, "**/lumi.conf"), recursive = True)
    sub_dirs = glob.glob(os.path.join(indir, '*/'))
    event_file_candidates = []
    for sub_dir in sub_dirs:
        eventfile_path = os.path.join(sub_dir, eventfile[channel])

        # ignore any subdirectory that does not have a lumi file in it
        if IsGoodLumiFile(os.path.join(sub_dir, "lumi.conf")) and IsGoodEventFile(eventfile_path):
            event_file_candidates.append(eventfile_path)
        else:
            print("Warning: '{}' does not have a good lumi file or a corrupted event file, ignoring its events!".format(sub_dir))

    print("have found {} good event files in this directory".format(len(event_file_candidates)))

    # combine them together
    output_file = os.path.join(indir, "events.h5")
    h5add(output_file, event_file_candidates)

if __name__ == "__main__":
    parser = ArgumentParser(description = "combine event files")
    parser.add_argument("indir", nargs = '+', action = "store")
    parser.add_argument("--channel", action = "store", default = "0lep")
    args = vars(parser.parse_args())

    CombineEventFiles(**args)
