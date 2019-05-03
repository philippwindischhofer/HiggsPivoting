import os, argparse, pickle
from argparse import ArgumentParser

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalAsimovPlots(model_dirs, plot_dir):
    # first, load the HistFitter output and also the corresponding sensdicts
    hypodicts = []
    sensdicts = []

    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "hypodict.pkl"), "rb") as fit_infile, open(os.path.join(model_dir, "sensdict.pkl"), "rb") as sens_infile:
                hypodict = pickle.load(fit_infile)
                sensdict = pickle.load(sens_infile)
                hypodicts.append(hypodict)
                sensdicts.append(sensdict)
        except:
            print("either sensdict.pkl or hypodict.pkl not found for model '{}'".format(model_dir))

    # now have all the data, just need to plot it
    PerformancePlotter.plot_asimov_significance_comparison(hypodicts, sensdicts, outdir = plot_dir)

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
