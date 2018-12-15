import uproot as ur
import pandas as pd

def raw_data(file_path, tree_name, branches):
    tree = ur.open(file_path)[tree_name]

    for chunk in tree.iterate(branches):
        converted = {bytes.decode(key): val for key, val in chunk.items()} # convert the column names from byte arrays into proper strings
        yield pd.DataFrame(converted)
