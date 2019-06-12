import os, pickle
import numpy as np
from argparse import ArgumentParser

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalCBAOptimizationPlots(opt_dirs, plot_dir):
    
    # for each directory, load the final optimization result and compare them
    optdicts = []
    evoldicts = []
    for opt_dir in opt_dirs:
        with open(os.path.join(opt_dir, "opt_results.pkl"), "rb") as infile:
            optdict = pickle.load(infile)
            optdicts.append(optdict)
        with open(os.path.join(opt_dir, "fit_evolution.pkl"), "rb") as infile:
            evoldict = pickle.load(infile)
            evoldicts.append(evoldict)

    # sort them and print an overview
    optdicts_sorted = sorted(optdicts, key = lambda cur: cur["target"])
    for cur in optdicts_sorted:
        print("{} --> {} sigma".format(cur["params"], cur["target"]))

    # plot the evolution over time of the optimization, for all available runs
    evolcurves = []
    stepcurves = []
    for evoldict in evoldicts:
        cur_stepcurve = sorted(evoldict.keys())
        cur_curve = [evoldict[pos]["combined"] for pos in cur_stepcurve]

        cur_curve = np.array(cur_curve)
        cur_curve = np.maximum.accumulate(cur_curve)
        cur_stepcurve = np.array(cur_stepcurve)

        evolcurves.append(cur_curve)
        stepcurves.append(cur_stepcurve)

    all_evolcurves = np.array(evolcurves)

    # compute the median, as well as the position of the upper and lower uncertainty bands
    median = np.median(all_evolcurves, axis = 0)
    unc_up = np.max(all_evolcurves, axis = 0) - median
    unc_down = -(median - np.min(all_evolcurves, axis = 0))

    PerformancePlotter._uncertainty_plot(stepcurves[0], median, 
                                         unc_up = unc_up, unc_down = unc_down, label = "Bayesian Optimization", outfile = os.path.join(plot_dir, "evolution.pdf"), xlabel = "step", ylabel = "Asimov significance", color = "black")

if __name__ == "__main__":
    parser = ArgumentParser(description = "create global summary plots for CBA optimization")
    parser.add_argument("--plot_dir")
    parser.add_argument("opt_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    MakeGlobalCBAOptimizationPlots(**args)
