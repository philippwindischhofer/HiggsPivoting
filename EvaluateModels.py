import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from argparse import ArgumentParser

from models.AdversarialEnvironment import AdversarialEnvironment
from training.ModelEvaluator import ModelEvaluator
from plotting.TrainingStatisticsPlotter import TrainingStatisticsPlotter
from plotting.PerformancePlotter import PerformancePlotter

def main():
    parser = ArgumentParser(description = "evaluate adversarial networks")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--plot_dir", action = "store", dest = "plot_dir")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    infile_path = args["infile_path"]
    model_dirs = args["model_dirs"]
    plot_dir = args["plot_dir"]

    # read the training data
    print("loading data ...")
    sig_data, bkg_data = pd.read_hdf(infile_path, key = 'Hbb'), pd.read_hdf(infile_path, key = 'Zbb')
    print("got " + str(len(sig_data)) + " signal events")
    print("got " + str(len(bkg_data)) + " background events")

    test_size = 0.2
    sig_data_train, sig_data_test = train_test_split(sig_data, test_size = test_size, random_state = 12345)
    bkg_data_train, bkg_data_test = train_test_split(bkg_data, test_size = test_size, random_state = 12345)

    mods = []
    perfdicts = []
    for model_dir in model_dirs:
        print("now evaluating " + model_dir)

        mce = AdversarialEnvironment.from_file(model_dir)
        mods.append(mce)

        #plots_outdir = os.path.join(plot_dir, os.path.basename(os.path.normpath(model_dir)))
        plots_outdir = plot_dir

        # generate performance plots for each model individually
        ev = ModelEvaluator(mce)
        ev.performance_plots(sig_data_test, bkg_data_test, plots_outdir)

        # get performance metrics and save them
        perfdict = ev.get_performance_metrics(sig_data_test, bkg_data_test)
        perfdicts.append(perfdict)
        print("got perfdict = " + str(perfdict))
        with open(os.path.join(plots_outdir, "perfdict.pkl"), "wb") as outfile:
            pickle.dump(perfdict, outfile)

        # generate plots showing the evolution of certain parameters during training
        tsp = TrainingStatisticsPlotter(model_dir)
        tsp.plot(outdir = plots_outdir)

    # also get the purely data-based performance measures that are available
    # datadict = ModelEvaluator.get_data_metrics(sig_data, bkg_data)
    # perfdicts.append(datadict)

    # generate combined performance plots that compare all the models
    # PerformancePlotter.plot(perfdicts, outpath = plot_dir)

if __name__ == "__main__":
    main()
