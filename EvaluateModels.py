import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from argparse import ArgumentParser

from models.AdversarialEnvironment import AdversarialEnvironment
from plotting.ModelEvaluator import ModelEvaluator
from plotting.TrainingStatisticsPlotter import TrainingStatisticsPlotter
from plotting.PerformancePlotter import PerformancePlotter
from DatasetExtractor import TrainNuisAuxSplit
from base.Configs import TrainingConfig

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
    sig_samples = ["Hbb"]
    bkg_samples = ["ttbar", "Zjets", "Wjets", "diboson", "singletop"]

    print("loading data ...")
    sig_data = [pd.read_hdf(infile_path, key = sig_sample) for sig_sample in sig_samples]
    bkg_data = [pd.read_hdf(infile_path, key = bkg_sample) for bkg_sample in bkg_samples]

    # extract the test dataset
    test_size = TrainingConfig.test_size
    sig_data_test = []
    sig_mBB_test = []
    sig_dRBB_test = []
    sig_pTB1_test = []
    sig_pTB2_test = []
    sig_weights_test = []
    for sample in sig_data:
        _, cur_test = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights
        cur_dRBBdata = cur_test[["dRBB"]].values
        cur_pTB1data = cur_test[["pTB1"]].values
        cur_pTB2data = cur_test[["pTB2"]].values
        sig_data_test.append(cur_testdata)
        sig_mBB_test.append(cur_nuisdata)
        sig_weights_test.append(cur_weights)
        sig_dRBB_test.append(cur_dRBBdata)
        sig_pTB1_test.append(cur_pTB1data)
        sig_pTB2_test.append(cur_pTB2data)

    bkg_data_test = []
    bkg_mBB_test = []
    bkg_dRBB_test = []
    bkg_pTB1_test = []
    bkg_pTB2_test = []
    bkg_weights_test = []
    for sample in bkg_data:
        _, cur_test = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights
        cur_dRBBdata = cur_test[["dRBB"]].values
        cur_pTB1data = cur_test[["pTB1"]].values
        cur_pTB2data = cur_test[["pTB2"]].values
        bkg_data_test.append(cur_testdata)
        bkg_mBB_test.append(cur_nuisdata)
        bkg_weights_test.append(cur_weights)
        bkg_dRBB_test.append(cur_dRBBdata)
        bkg_pTB1_test.append(cur_pTB1data)
        bkg_pTB2_test.append(cur_pTB2data)

    for model_dir in model_dirs:
        print("now evaluating " + model_dir)

        mce = AdversarialEnvironment.from_file(model_dir)
        plots_outdir = plot_dir

        # generate performance plots for each model individually
        ev = ModelEvaluator(mce)

        # plot the output distribution of the classifier for a few events from each sample
        for sample, sample_name in zip(sig_data_test + bkg_data_test, sig_samples + bkg_samples):
            for event_num in range(10):
                event = sample[[event_num]]
                ev.plot_clf_pdf(event = event, varlabels = TrainingConfig.training_branches, plotlabel = sample_name, outpath = os.path.join(plots_outdir, "clf_pdf_" + sample_name + "_" + str(event_num) + ".pdf"))

        ev.plot_roc(data_sig = sig_data_test, data_bkg = bkg_data_test, sig_weights = sig_weights_test, bkg_weights = bkg_weights_test, outpath = plots_outdir)

        # generate distortion plots
        ev.plot_distortion(data_sig = sig_data_test, data_bkg = bkg_data_test, var_sig = sig_mBB_test, var_bkg = bkg_mBB_test, 
                           weights_sig = sig_weights_test, weights_bkg = bkg_weights_test, sigeffs = [1.0, 0.5, 0.25], outpath = plots_outdir, 
                           labels_sig = sig_samples, labels_bkg = bkg_samples, xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.", path_prefix = "dist_mBB")
        ev.plot_distortion(data_sig = sig_data_test, data_bkg = bkg_data_test, var_sig = sig_dRBB_test, var_bkg = bkg_dRBB_test, 
                           weights_sig = sig_weights_test, weights_bkg = bkg_weights_test, sigeffs = [1.0, 0.5, 0.25], outpath = plots_outdir, 
                           labels_sig = sig_samples, labels_bkg = bkg_samples, xlabel = r'$\Delta R_{bb}$', ylabel = "a.u.", path_prefix = "dist_dRBB", histrange = (0, 5))
        ev.plot_distortion(data_sig = sig_data_test, data_bkg = bkg_data_test, var_sig = sig_pTB1_test, var_bkg = bkg_pTB1_test, 
                           weights_sig = sig_weights_test, weights_bkg = bkg_weights_test, sigeffs = [1.0, 0.5, 0.25], outpath = plots_outdir, 
                           labels_sig = sig_samples, labels_bkg = bkg_samples, xlabel = r'$p_{T, b(1)}$ [GeV]', ylabel = "a.u.", path_prefix = "dist_pTB1")
        ev.plot_distortion(data_sig = sig_data_test, data_bkg = bkg_data_test, var_sig = sig_pTB2_test, var_bkg = bkg_pTB2_test, 
                           weights_sig = sig_weights_test, weights_bkg = bkg_weights_test, sigeffs = [1.0, 0.5, 0.25], outpath = plots_outdir, 
                           labels_sig = sig_samples, labels_bkg = bkg_samples, xlabel = r'$p_{T, b(2)}$ [GeV]', ylabel = "a.u.", path_prefix = "dist_pTB2")

        # plot classifier distributions
        ev.plot_clf_distribution(data = sig_data_test + bkg_data_test, weights = sig_weights_test + bkg_weights_test, outpath = plots_outdir, labels = sig_samples + bkg_samples, num_cols = 2)

        # plot correlation plots of the classifier with m_BB
        ev.plot_clf_correlations(varname = "mBB", data_sig = sig_data_test, weights_sig = sig_weights_test, labels_sig = sig_samples, data_bkg = bkg_data_test, weights_bkg = bkg_weights_test, labels_bkg = bkg_samples, outpath = plots_outdir)

        # get performance metrics and save them
        perfdict = ev.get_performance_metrics(sig_data_test, bkg_data_test, sig_mBB_test, bkg_mBB_test, sig_weights_test, 
                                              bkg_weights_test, labels_sig = sig_samples, labels_bkg = bkg_samples)
        print("got perfdict = " + str(perfdict))
        with open(os.path.join(plots_outdir, "perfdict.pkl"), "wb") as outfile:
           pickle.dump(perfdict, outfile)

        # generate plots showing the evolution of certain parameters during training
        tsp = TrainingStatisticsPlotter(model_dir)
        tsp.plot(outdir = plots_outdir)

if __name__ == "__main__":
    main()
