import os
import pandas as pd
from argparse import ArgumentParser

def filter_callback(table):
    return table[table["mBB"] < 250.0]

def h5filter(outfile, infile):
    tables = {}

    # check which keys (i.e. tables) are available in the present input file
    with pd.HDFStore(infile) as hdf:
        keys = hdf.keys()
        available_tables = [os.path.basename(key) for key in keys]

    # then, read in the tables present in this file
    for cur_table_name in available_tables:
        cur_table = pd.read_hdf(infile, key = cur_table_name)
        filtered_table = filter_callback(cur_table)

        if os.path.exists(outfile):
            mode = 'a'
        else:
            mode = 'w'

        filtered_table.to_hdf(outfile, key = cur_table_name, mode = mode)

if __name__ == "__main__":
    parser = ArgumentParser("behaves like hadd, but for h5 files")
    parser.add_argument("--in", action = "store", dest = "infile")
    parser.add_argument("--out", action = "store", dest = "outfile")
    args = vars(parser.parse_args())

    h5filter(**args)
