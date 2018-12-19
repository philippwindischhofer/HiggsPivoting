import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from PCAWhiteningPreprocessor import PCAWhiteningPreprocessor
from SimpleModel import SimpleModel
from SimpleClassifierEnvironment import SimpleClassifierEnvironment
from ModelEvaluator import ModelEvaluator

from Configs import TrainingConfig

def main():
    infile_path = "/data/atlas/atlasdata/windischhofer/Hbb/training-mc16d.h5"
    model_dir = "/home/windischhofer/HiggsPivoting/models"

    # read the training data
    print("loading data ...")
    sig_data, bkg_data = pd.read_hdf(infile_path, key = 'sig'), pd.read_hdf(infile_path, key = 'bkg')
    print("got " + str(len(sig_data)) + " signal events")
    print("got " + str(len(bkg_data)) + " background events")

    test_size = 0.2
    sig_data_train, sig_data_test = train_test_split(sig_data, test_size = test_size)
    bkg_data_train, bkg_data_test = train_test_split(bkg_data, test_size = test_size)

    # load the trained model
    mod = SimpleModel("test_model", hyperpars = {"num_hidden_layers": 3, "num_units": 30})
    sce = SimpleClassifierEnvironment(classifier_model = mod)
    sce.build(num_inputs = len(TrainingConfig.training_branches))
    sce.load(os.path.join(model_dir, "test_model.dat"))

    # generate performance plots
    ev = ModelEvaluator(sce)
    ev.evaluate(sig_data_test, bkg_data_test, "/home/windischhofer/HiggsPivotingModels/")

if __name__ == "__main__":
    main()
