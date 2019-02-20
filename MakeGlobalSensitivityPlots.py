import os, argparse
from argparse import ArgumentParser

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalSensitivityPlots(model_dirs, plot_dir):
    pass

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
