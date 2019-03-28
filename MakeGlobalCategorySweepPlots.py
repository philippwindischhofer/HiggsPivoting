import os, argparse, pickle
from argparse import ArgumentParser

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalCategorySweepPlots(model_dirs, plot_dir):
    # first, load the HistFitter output and also the corresponding sensdicts
    hypodicts = []
    catdicts = []

    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "hypodict.pkl"), "rb") as fit_infile, open(os.path.join(model_dir, "catdict.pkl"), "rb") as cat_infile:
                hypodict = pickle.load(fit_infile)
                catdict = pickle.load(cat_infile)
                hypodicts.append(hypodict)
                catdicts.append(catdict)
        except:
            print("either sensdict.pkl or catdict.pkl not found for model '{}'".format(model_dir))

    asimov_sig_names = ["asimov_sig_background_floating", "asimov_sig_background_fixed"]

    # now have all the data, just need to plot it
    for asimov_sig_name in asimov_sig_names:
        PerformancePlotter.plot_asimov_significance_category_sweep_comparison(hypodicts, catdicts, outfile = os.path.join(plot_dir, asimov_sig_name + "_comparison.pdf"), asimov_sig_name = asimov_sig_name)

def main():
    parser = ArgumentParser(description = "create global summary plots for Asimov sensitivities")
    parser.add_argument("--plotdir")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    plot_dir = args["plotdir"]
    model_dirs = args["model_dirs"]

    MakeGlobalCategorySweepPlots(model_dirs, plot_dir)

if __name__ == "__main__":
    main()
