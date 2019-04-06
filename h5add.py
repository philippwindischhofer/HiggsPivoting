import os
import pandas as pd
from argparse import ArgumentParser

def h5add(output_file, input_files):
    tables = {}

    for input_file in input_files:
        # check which keys (i.e. tables) are available in the present input file
        with pd.HDFStore(input_file) as hdf:
            keys = hdf.keys()
            available_tables = [os.path.basename(key) for key in keys]

        # then, read in the tables present in this file
        for cur_table_name in available_tables:
            cur_table = pd.read_hdf(input_file, key = cur_table_name)
            
            if not cur_table_name in tables:
                tables[cur_table_name] = []
            tables[cur_table_name].append(cur_table)

        # now, concatenate all the read tables and dump them into the output file
        for cur_table_name in tables.keys():
            merged_table = pd.concat(tables[cur_table_name]).reset_index(drop = True)
            
            if os.path.exists(output_file):
                mode = 'a'
            else:
                mode = 'w'

            merged_table.to_hdf(output_file, key = cur_table_name, mode = mode)

if __name__ == "__main__":
    parser = ArgumentParser("behaves like hadd, but for h5 files")
    parser.add_argument("files", nargs = "+", action = "store")
    args = vars(parser.parse_args())

    files = args["files"]
    output_file = files[0]
    input_files = files[1:]

    if os.path.exists(output_file):
        raise Exception("Error: output file already exists!")

    h5add(output_file, input_files)
