import os, argparse, pickle
from argparse import ArgumentParser

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalSensitivityPlots(model_dirs, plot_dir):
    sensdicts = []

    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "sensdict.pkl"), "rb") as infile:
                sensdict = pickle.load(infile)
                sensdicts.append(sensdict)
        except:
            print("no information found for model '{}'".format(model_dir))

    # generate combined sensitivity plots that show all models
    PerformancePlotter.plot_significance_KS(sensdicts, outfile = os.path.join(plot_dir, "sensitivity_KS.pdf"))

def main():
    parser = ArgumentParser(description = "create global sensitivity plots")
    parser.add_argument("--plotdir")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    plot_dir = args["plotdir"]
    model_dirs = args["model_dirs"]

    MakeGlobalSensitivityPlots(model_dirs, plot_dir)

if __name__ == "__main__":
    main()
