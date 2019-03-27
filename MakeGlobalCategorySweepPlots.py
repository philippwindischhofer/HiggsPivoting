import os, argparse, pickle
from argparse import ArgumentParser

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalCategorySweepPlots(model_dirs, plot_dir):
    # first, load the HistFitter output and also the corresponding sensdicts
    hypodicts = []
    perfdicts = []

    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "hypodict.pkl"), "rb") as fit_infile, open(os.path.join(model_dir, "catdict.pkl"), "rb") as cat_infile, open(os.path.join(model_dir, "perfdict.pkl"), "rb") as perf_infile:
                hypodict = pickle.load(fit_infile)
                perfdict = pickle.load(perf_infile)
                hypodicts.append(hypodict)
                perfdicts.append(perfdict)
        except:
            print("either sensdict.pkl or hypodict.pkl not found for model '{}'".format(model_dir))

    # now have all the data, just need to plot it
    PerformancePlotter.plot_asimov_significance_comparison(hypodicts, perfdicts, outfile = os.path.join(plot_dir, "asimov_binned_significance_comparison.pdf"))

def main():
    parser = ArgumentParser(description = "create global summary plots for Asimov sensitivities")
    parser.add_argument("--plotdir")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    plot_dir = args["plotdir"]
    model_dirs = args["model_dirs"]

    MakeGlobalAsimovPlots(model_dirs, plot_dir)

if __name__ == "__main__":
    main()
