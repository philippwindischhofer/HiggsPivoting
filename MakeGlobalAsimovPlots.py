import os, argparse, pickle
from argparse import ArgumentParser

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalAsimovPlots(model_dirs, plot_dir):
    # first, load the HistFitter output and also the corresponding sensdicts
    pardicts = []
    sensdicts = []

    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "pardict.pkl"), "rb") as fit_infile, open(os.path.join(model_dir, "sensdict.pkl"), "rb") as sens_infile:
                pardict = pickle.load(fit_infile)
                sensdict = pickle.load(sens_infile)
                pardicts.append(pardict)
                sensdicts.append(sensdict)
        except:
            print("either sensdict.pkl or pardict.pkl not found for model '{}'".format(model_dir))

    # now have all the data, just need to plot it
    PerformancePlotter.plot_asimov_binned_significance(pardicts, sensdicts, outfile = os.path.join(plot_dir, "asimov_binned_significance_comparison.pdf"))

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
