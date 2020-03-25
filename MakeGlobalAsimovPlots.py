import os, argparse, pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from plotting.PerformancePlotter import PerformancePlotter

def load_plotdata(model_dirs, lambda_veto = lambda cur: False):
    hypodicts = []
    sensdicts = []

    model_performance = {}

    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "hypodict.pkl"), "rb") as fit_infile, open(os.path.join(model_dir, "anadict.pkl"), "rb") as sens_infile:
                hypodict = pickle.load(fit_infile)
                sensdict = pickle.load(sens_infile)
                if lambda_veto(float(sensdict["lambda"])):
                    continue

                hypodicts.append(hypodict)
                sensdicts.append(sensdict)

                cur_dir = os.path.split(model_dir)[-1]
                model_performance[cur_dir] = hypodict["asimov_sig_ncat_background_floating"]
        except:
            print("either sensdict.pkl or perfdict.pkl not found for model '{}'".format(model_dir))

    # sort and print the list of Asimov significances achieved by each model
    sorted_model_performance = sorted(model_performance.items(), key = lambda cur: cur[1])

    for (name, sig) in sorted_model_performance:
        print("{}: {} sigma".format(name, sig))

    print("CBA original: {} sigma".format(hypodict["original_asimov_sig_high_low_MET_background_floating"]))
    print("CBA optimized: {} sigma".format(hypodict["optimized_asimov_sig_high_low_MET_background_floating"]))

    return hypodicts, sensdicts

def MakeGlobalAsimovPlots(model_dirs, plot_dir):

    hypodicts, sensdicts = load_plotdata(model_dirs)

    # now have all the data, just need to plot it
    dark_blue = plt.cm.Blues(1000)
    PerformancePlotter.plot_asimov_significance_comparison([hypodicts], [sensdicts], colors = [dark_blue], labels = ["pivotal classifier"], outdir = plot_dir, plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s}=13$ TeV, 140 fb$^{-1}$'])

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
