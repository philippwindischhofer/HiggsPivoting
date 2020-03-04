import os
from argparse import ArgumentParser
from MakeGlobalAsimovPlots import load_plotdata
import matplotlib.pyplot as plt

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalAsimovComparisonPlots(plotdir, workdirs, labels):
    
    hypodicts = []
    sensdicts = []

    cmaps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges]
    colors = [cmap(0.7) for cmap in cmaps]

    lambda_upper_limits = {"MIND": 0.18, "DisCo": 1.2, "EMAX": 1.8}

    for workdir, label in zip(workdirs, labels):
        cur_upper_limit = lambda_upper_limits[label]

        cur_model_dirs = filter(os.path.isdir, map(lambda cur: os.path.join(workdir, cur), os.listdir(workdir)))
        cur_hypodicts, cur_sensdicts = load_plotdata(cur_model_dirs, lambda_upper_limit = cur_upper_limit)

        lambdas = [float(cur_dict["lambda"]) for cur_dict in cur_sensdicts]
        lambda_max = max(lambdas)

        for cur_sensdict in cur_sensdicts:
            cur_sensdict["lambda"] = str(float(cur_sensdict["lambda"]) / lambda_max)

        hypodicts.append(cur_hypodicts)
        sensdicts.append(cur_sensdicts)

    PerformancePlotter.plot_asimov_significance_comparison(hypodicts, sensdicts, colors = colors, labels = labels, outdir = plotdir, plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s}=13$ TeV, 140 fb$^{-1}$'],
                                                           xlabel = r"$\lambda / \lambda_{\mathrm{max}}$")

if __name__ == "__main__":
    parser = ArgumentParser(description = "create global comparison plots for Asimov sensitivities")
    parser.add_argument("--plotdir", action = "store")
    parser.add_argument("--workdirs", nargs = '+', action = "store")
    parser.add_argument("--labels", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    MakeGlobalAsimovComparisonPlots(**args)
