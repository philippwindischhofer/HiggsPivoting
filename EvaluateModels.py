import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from PCAWhiteningPreprocessor import PCAWhiteningPreprocessor
from SimpleModel import SimpleModel
from ModelEvaluator import ModelEvaluator
from TrainingStatisticsPlotter import TrainingStatisticsPlotter

from MINEClassifierEnvironment import MINEClassifierEnvironment
from AdversarialClassifierEnvironment import AdversarialClassifierEnvironment

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
    for model_dir in model_dirs:
        print("now evaluating " + model_dir)

        # load the trained models
        mod = SimpleModel("test_model", hyperpars = {"num_hidden_layers": 2, "num_units": 30})
        #mce = MINEClassifierEnvironment(classifier_model = mod)
        mce = AdversarialClassifierEnvironment(classifier_model = mod)
        mce.build(num_inputs = len(TrainingConfig.training_branches), num_nuisances = 1, lambda_val = 0.6)
        mce.load(os.path.join(model_dir, "model.dat"))
        mods.append(mce)

        plots_outdir = os.path.join(plot_dir, os.path.basename(os.path.normpath(model_dir)))
        # generate performance plots for each model individually
        ev = ModelEvaluator(mce)
        ev.evaluate(sig_data_test, bkg_data_test, plots_outdir)

        # generate plots showing the evolution of certain parameters during training
        tsp = TrainingStatisticsPlotter(model_dir)
        tsp.plot(outdir = plots_outdir)

    # generate combined performance plots that compare all the models
    

if __name__ == "__main__":
    main()
