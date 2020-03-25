import os
from argparse import ArgumentParser
from MakeGlobalAsimovPlots import load_plotdata
import matplotlib.pyplot as plt

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalAsimovComparisonPlots(plotdir, workdirs, labels):
    
    hypodicts = []
    sensdicts = []

    cmaps = [plt.cm.Greens, plt.cm.Blues, plt.cm.Oranges]
    colors = [cmap(0.7) for cmap in cmaps]
    xlabels = []

    def MIND_veto(lambda_val):
        if lambda_val > 0.0 and lambda_val < 0.5:
            return True
        else:
            return False
        return False

    def DisCo_veto(lambda_val):
        return False
            
    def EMAX_veto(lambda_val):
        if lambda_val > 0.0 and lambda_val < 1.5:
            return True
        else:
            return False
        return False

    lambda_veto = {"MIND": MIND_veto, "DisCo": DisCo_veto, "EMAX": EMAX_veto}
    linthresh_library = {"MIND": 3.9, "DisCo": 3.4, "EMAX": 9.7}

    linthreshs = []

    for workdir, label in zip(workdirs, labels):
        if label in lambda_veto:
            cur_veto = lambda_veto[label]
        else:
            cur_veto = lambda cur: False

        xlabels.append(r"$\lambda_{{\mathrm{{{}}}}}$".format(label))

        cur_model_dirs = filter(os.path.isdir, map(lambda cur: os.path.join(workdir, cur), os.listdir(workdir)))
        cur_hypodicts, cur_sensdicts = load_plotdata(cur_model_dirs, lambda_veto = cur_veto)

        hypodicts.append(cur_hypodicts)
        sensdicts.append(cur_sensdicts)
        linthreshs.append(linthresh_library[label])

    PerformancePlotter.plot_asimov_significance_comparison(hypodicts, sensdicts, colors = colors, labels = labels, outdir = plotdir, plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s}=13$ TeV, 140 fb$^{-1}$'],
                                                           xlabel = xlabels, linthreshs = linthreshs)

if __name__ == "__main__":
    parser = ArgumentParser(description = "create global comparison plots for Asimov sensitivities")
    parser.add_argument("--plotdir", action = "store")
    parser.add_argument("--workdirs", nargs = '+', action = "store")
    parser.add_argument("--labels", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    MakeGlobalAsimovComparisonPlots(**args)
