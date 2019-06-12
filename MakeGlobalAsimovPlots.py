import os, argparse, pickle
from argparse import ArgumentParser

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalAsimovPlots(model_dirs, plot_dir):
    # load the pre-optimization CBA values
    CBA_pre_opt_path = "/home/windischhofer/datasmall/Hbb_adv_madgraph_2019_06_10/MINERandomizedClassifierATLAS/Master_slice_0.0"
    with open(os.path.join(CBA_pre_opt_path, "hypodict.pkl"), "rb") as fit_infile:
        CBA_pre_opt = pickle.load(fit_infile)

    # load the HistFitter output and also the corresponding sensdicts
    hypodicts = []
    sensdicts = []

    model_performance = {}

    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "hypodict.pkl"), "rb") as fit_infile, open(os.path.join(model_dir, "perfdict.pkl"), "rb") as sens_infile:
                hypodict = pickle.load(fit_infile)
                sensdict = pickle.load(sens_infile)
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

    print("CBA: {} sigma".format(hypodict["asimov_sig_high_low_MET_background_floating"]))

    # now have all the data, just need to plot it
    PerformancePlotter.plot_asimov_significance_comparison(hypodicts, sensdicts, outdir = plot_dir, plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s}=13$ TeV, 140 fb$^{-1}$'],
                                                           CBA_overlays = [{"label": "cut-based analysis", "ls": "--", "dict": CBA_pre_opt},
                                                                           {"label": "cut-based analysis (re-optimized)", "ls": ":", "dict": hypodict}])

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
