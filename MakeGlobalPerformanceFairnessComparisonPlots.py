import os, pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from plotting.PerformancePlotter import PerformancePlotter

def load_plotdata(model_dirs, lambda_upper_limit):
    dicts = []

    # load back the prepared performance metrics
    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "anadict.pkl"), "rb") as infile:
                anadict = pickle.load(infile)
                if float(anadict["lambda"]) > lambda_upper_limit:
                    continue

                dicts.append(anadict)
        except:
            print("no information found for model '{}'".format(model_dir))

    dicts = sorted(dicts, key = lambda cur: cur["lambda"])
    return dicts

def MakeGlobalPerformanceFairnessComparisonPlots(plotdir, workdirs, labels):

    # load the full collection of data to be plotted, for each model directory
    dicts = []
    colorschemes = []
    colorscheme_library = [plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges]

    lambda_upper_limits = {"MIND": 1e6, "DisCo": 1e6, "EMAX": 1e6}

    for workdir, cur_colorscheme, cur_label in zip(workdirs, colorscheme_library, labels):
        cur_upper_limit = lambda_upper_limits[cur_label]
        cur_model_dirs = filter(os.path.isdir, map(lambda cur: os.path.join(workdir, cur), os.listdir(workdir)))
        cur_dicts = load_plotdata(cur_model_dirs, cur_upper_limit)
        colorschemes.append(cur_colorscheme)
        dicts.append(cur_dicts)

    PerformancePlotter.plot_significance_fairness_combined_legend(dicts, colorschemes, plotdir, series_labels = labels)
    PerformancePlotter.plot_significance_fairness_combined(dicts, colorschemes, plotdir, series_labels = labels, nJ = 2)
    PerformancePlotter.plot_significance_fairness_combined(dicts, colorschemes, plotdir, series_labels = labels, nJ = 3)
    PerformancePlotter.plot_significance_fairness_combined_smooth(dicts, colorschemes, plotdir, series_labels = labels, nJ = 2, show_legend = False)
    PerformancePlotter.plot_significance_fairness_combined_smooth(dicts, colorschemes, plotdir, series_labels = labels, nJ = 3, show_legend = False)

if __name__ == "__main__":
    parser = ArgumentParser(description = "make performance vs. fairness comparison plots")
    parser.add_argument("--plotdir")
    parser.add_argument("--workdirs", nargs = '+', action = "store")
    parser.add_argument("--labels", nargs = '+', action = "store", default = [])
    args = vars(parser.parse_args())

    MakeGlobalPerformanceFairnessComparisonPlots(**args)
