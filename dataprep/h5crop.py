import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser

def h5crop(infile, outfile, maxlen):
    tables = {}
    maxlen = int(maxlen)

    # check which keys (i.e. tables) are available in the present input file
    with pd.HDFStore(infile) as hdf:
        keys = hdf.keys()
        available_tables = [os.path.basename(key) for key in keys]

    # then, read in the tables present in this file
    for cur_table_name in available_tables:
        cur_table = pd.read_hdf(infile, key = cur_table_name)
        cur_table = cur_table.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample

        if len(cur_table) > maxlen:
            cropped_table = cur_table[:maxlen]
        else:
            cropped_table = cur_table

        # scale up the weights again
        cropped_table["EventWeight"] = cropped_table["EventWeight"] * np.sum(cur_table["EventWeight"]) / np.sum(cropped_table["EventWeight"])

        tables[cur_table_name] = cropped_table

    # now, concatenate all the read tables and dump them into the output file
    for cur_table_name, cur_table in tables.items():            

        if os.path.exists(outfile):
            mode = 'a'
        else:
            mode = 'w'
            
        cur_table.to_hdf(outfile, key = cur_table_name, mode = mode)

if __name__ == "__main__":
    parser = ArgumentParser("cuts down the length of tables stored in this file, but preserves the file structure")
    parser.add_argument("--infile", action = "store", dest = "infile")
    parser.add_argument("--outfile", action = "store", dest = "outfile")
    parser.add_argument("--maxlen", action = "store", dest = "maxlen")
    args = vars(parser.parse_args())

    if os.path.exists(args["outfile"]):
        raise Exception("Error: output file already exists!")

    h5crop(**args)
