import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from PCAWhiteningPreprocessor import PCAWhiteningPreprocessor
from SimpleModel import SimpleModel
from ModelEvaluator import ModelEvaluator
from TrainingStatisticsPlotter import TrainingStatisticsPlotter
from PerformancePlotter import PerformancePlotter

from ConfigFileUtils import ConfigFileUtils
from Configs import TrainingConfig

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
    sig_data, bkg_data = pd.read_hdf(infile_path, key = 'sig'), pd.read_hdf(infile_path, key = 'bkg')
    print("got " + str(len(sig_data)) + " signal events")
    print("got " + str(len(bkg_data)) + " background events")

    test_size = 0.2
    sig_data_train, sig_data_test = train_test_split(sig_data, test_size = test_size, random_state = 12345)
    bkg_data_train, bkg_data_test = train_test_split(bkg_data, test_size = test_size, random_state = 12345)

    mods = []
    perfdicts = []
    for model_dir in model_dirs:
        print("now evaluating " + model_dir)

        model_type = ConfigFileUtils.get_env_type(model_dir)
        if model_type is not None:
            mce = model_type.from_file(model_dir)
        else:
            raise FileNotFoundError("could not determine the type of this model - aborting!")

        mods.append(mce)

        plots_outdir = os.path.join(plot_dir, os.path.basename(os.path.normpath(model_dir)))

        # generate performance plots for each model individually
        ev = ModelEvaluator(mce)
        ev.performance_plots(sig_data_test, bkg_data_test, plots_outdir)

        # get performance metrics
        perfdict = ev.get_performance_metrics(sig_data_test, bkg_data_test)
        perfdicts.append(perfdict)

        # generate plots showing the evolution of certain parameters during training
        tsp = TrainingStatisticsPlotter(model_dir)
        tsp.plot(outdir = plots_outdir)

    # generate combined performance plots that compare all the models
    PerformancePlotter.plot(perfdicts, outpath = plot_dir)

if __name__ == "__main__":
    main()
