import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np

def b_taggingWeight(col):
    return 0.85 * np.tanh(0.0025 * col) * (25.0 / (1 + 0.063 * col))

def ApplyTruthTaggingWeights(infile, outfile):
    # apply weights originating from the truth-tagging procedure:
    # weight individual events with their respective b-tagging efficiencies
    tables = {}

    with pd.HDFStore(infile) as hdf:
        keys = hdf.keys()
        available_tables = [os.path.basename(key) for key in keys]

    for cur_table_name in available_tables:
        cur_table = pd.read_hdf(infile, key = cur_table_name)
        tables[cur_table_name] = cur_table

    for table_name in tables.keys():
        tables[table_name]["EventWeight"] *= b_taggingWeight(tables[table_name]["pTB1"])
        tables[table_name]["EventWeight"] *= b_taggingWeight(tables[table_name]["pTB2"])

    # save them to the output file
    for table_name, table in tables.items():
        if os.path.exists(outfile):
            mode = 'a'
        else:
            mode = 'w'
            
        table.to_hdf(outfile, key = table_name, mode = mode)

if __name__ == "__main__":
    parser = ArgumentParser(description = "apply weights from truth tagging")
    parser.add_argument("--infile", action = "store", dest = "infile")
    parser.add_argument("--outfile", action = "store", dest = "outfile")
    args = vars(parser.parse_args())

    ApplyTruthTaggingWeights(**args)
