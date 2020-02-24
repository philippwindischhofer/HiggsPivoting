import os, pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from plotting.PerformancePlotter import PerformancePlotter

def load_plotdata(model_dirs):
    dicts = []

    # load back the prepared performance metrics
    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "anadict.pkl"), "rb") as infile:
                anadict = pickle.load(infile)
                dicts.append(anadict)
        except:
            print("no information found for model '{}'".format(model_dir))

    dicts = sorted(dicts, key = lambda cur: cur["lambda"])
    return dicts

def MakeGlobalPerformanceFairnessComparisonPlots(plotdir, workdirs, labels):

    # load the full collection of data to be plotted, for each model directory
    dicts = []
    colorschemes = []
    colorscheme_library = [plt.cm.Blues, plt.cm.YlGn]

    for workdir, cur_colorscheme in zip(workdirs, colorscheme_library):
        cur_model_dirs = filter(os.path.isdir, map(lambda cur: os.path.join(workdir, cur), os.listdir(workdir)))
        cur_dicts = load_plotdata(cur_model_dirs)
        colorschemes.append(cur_colorscheme)
        dicts.append(cur_dicts)
        
    PerformancePlotter.plot_significance_fairness_combined(dicts, colorschemes, plotdir, series_labels = labels, nJ = 2)
    PerformancePlotter.plot_significance_fairness_combined(dicts, colorschemes, plotdir, series_labels = labels, nJ = 3)

if __name__ == "__main__":
    parser = ArgumentParser(description = "make performance vs. fairness comparison plots")
    parser.add_argument("--plotdir")
    parser.add_argument("--workdirs", nargs = '+', action = "store")
    parser.add_argument("--labels", nargs = '+', action = "store", default = [])
    args = vars(parser.parse_args())

    MakeGlobalPerformanceFairnessComparisonPlots(**args)
