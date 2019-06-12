import os, pickle
from argparse import ArgumentParser

def MakeGlobalCBAOptimizationPlots(opt_dirs, plot_dir):
    
    # for each directory, load the final optimization result and compare them
    optdicts = []
    for opt_dir in opt_dirs:
        with open(os.path.join(opt_dir, "opt_results.pkl"), "rb") as infile:
            optdict = pickle.load(infile)
            optdicts.append(optdict)

    # sort them
    optdicts_sorted = sorted(optdicts, key = lambda cur: cur["target"])
    for cur in optdicts_sorted:
        print("{} --> {} sigma".format(cur["params"], cur["target"]))

if __name__ == "__main__":
    parser = ArgumentParser(description = "create global summary plots for CBA optimization")
    parser.add_argument("--plot_dir")
    parser.add_argument("opt_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    MakeGlobalCBAOptimizationPlots(**args)
