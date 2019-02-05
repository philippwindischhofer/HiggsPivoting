import os, pickle
from argparse import ArgumentParser

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalPlots(model_dirs, plot_dir):
    perfdicts = []

    # load back the prepared performance metrics
    for model_dir in model_dirs:
        with open(os.path.join(model_dir, "perfdict.pkl"), "rb") as infile:
            perfdict = pickle.load(infile)
        perfdicts.append(perfdict)

    # generate combined performance plots that compare all the models
    PerformancePlotter.plot(perfdicts, outpath = plot_dir, colorquant = "lambda")

if __name__ == "__main__":
    parser = ArgumentParser(description = "create global comparison plots")
    parser.add_argument("--plotdir")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    plot_dir = args["plotdir"]
    model_dirs = args["model_dirs"]

    MakeGlobalPlots(model_dirs, plot_dir)
