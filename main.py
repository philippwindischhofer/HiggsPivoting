import tensorflow as tf
import numpy as np
import uproot as ur
import pandas as pd

def raw_generator(file_path, tree_name, branches):
    tree = ur.open(file_path)[tree_name]

    for chunk in tree.iterate(branches):
        converted = {bytes.decode(key): val for key, val in chunk.items()} # convert the column names from byte arrays into proper strings
        yield pd.DataFrame(converted)
        
def main():
    file_path = "/data/atlas/atlasdata/windischhofer/Hbb/hist-all-mc16d.root"

    data_branches = ["nTags"]
    truth_branches = ["Sample"]
    read_branches = data_branches + truth_branches

    sig_samples = ["Wl"]
    bkg_samples = ["asdf"]

    # convert them into binary representations, to match the way they are read from the tree
    sig_samples_bin = [str.encode(samp) for samp in sig_samples]
    bkg_samples_bin = [str.encode(samp) for samp in bkg_samples]

    test = raw_generator(file_path, "Nominal", read_branches)

    for chunk in test:
        data_chunk = chunk[data_branches]
        truth_chunk = chunk[truth_branches]

        bkg_chunk = chunk.loc[truth_chunk["Sample"].isin(bkg_samples_bin)]
        sig_chunk = chunk.loc[truth_chunk["Sample"].isin(sig_samples_bin)]

        print(bkg_chunk)
        print(sig_chunk)

if __name__ == "__main__":
    main()
