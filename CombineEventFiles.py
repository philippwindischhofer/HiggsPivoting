import os, glob
from argparse import ArgumentParser
from h5add import h5add

def CombineEventFiles(indir):
    assert len(indir) == 1
    indir = indir[0]

    # first, look for all existing lumi files
    # Note: this semi-automatic way of doing it is faster than simply
    #     lumifiles = glob.glob(os.path.join(indir, "**/lumi.conf"), recursive = True)
    sub_dirs = glob.glob(os.path.join(indir, '*/'))
    event_file_candidates = []
    for sub_dir in sub_dirs:

        # ignore any subdirectory that does not have a lumi file in it
        if not os.path.isfile(os.path.join(sub_dir, "lumi.conf")):
            continue

        eventfile_paths = glob.glob(os.path.join(sub_dir, "*.h5"))
        for eventfile_path in eventfile_paths:
            if os.path.isfile(eventfile_path):
                event_file_candidates.append(eventfile_path)

    print("have found {} event files in this directory".format(len(event_file_candidates)))

    # combine them together
    output_file = os.path.join(indir, "events.h5")
    h5add(output_file, event_file_candidates)

if __name__ == "__main__":
    parser = ArgumentParser(description = "combine event files")
    parser.add_argument("indir", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    CombineEventFiles(**args)
