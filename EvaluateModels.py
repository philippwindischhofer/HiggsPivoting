import os, pickle
import numpy as np
import pandas as pd
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
    parser.add_argument("--plot_clf_distribs", action = "store_const", const = True, default = False)
    parser.add_argument("--use_test", action = "store_const", const = True, default = False)
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    infile_path = args["infile_path"]
    model_dirs = args["model_dirs"]
    plot_dir = args["plot_dir"]
    plot_clf_distribs = args["plot_clf_distribs"]

    # read the training data
    sig_samples = TrainingConfig.sig_samples
    bkg_samples = TrainingConfig.bkg_samples

    print("loading data ...")
    sig_data = [pd.read_hdf(infile_path, key = sig_sample) for sig_sample in sig_samples]
    bkg_data = [pd.read_hdf(infile_path, key = bkg_sample) for bkg_sample in bkg_samples]

    # extract the validation or test dataset
    if args["use_test"]:
        print("using test dataset")
        data_slice = TrainingConfig.test_slice
    else:
        print("using validation dataset")
        data_slice = TrainingConfig.validation_slice

    sig_data_test = []
    sig_data_test_2j = []
    sig_data_test_3j = []
    sig_aux_test = []
    sig_aux_test_2j = []
    sig_aux_test_3j = []
    sig_mBB_test = []
    sig_mBB_test_2j = []
    sig_mBB_test_3j = []
    sig_dRBB_test = []
    sig_dRBB_test_2j = []
    sig_dRBB_test_3j = []
    sig_pTB1_test = []
    sig_pTB2_test = []
    sig_weights_test = []
    sig_weights_test_2j = []
    sig_weights_test_3j = []
    for sample, sample_name in zip(sig_data, sig_samples):
        cur_length = len(sample)
        sample = sample.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample
        cur_test = sample[int(data_slice[0] * cur_length) : int(data_slice[1] * cur_length)]
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights
        cur_dRBBdata = cur_test[["dRBB"]].values
        cur_pTB1data = cur_test[["pTB1"]].values
        cur_pTB2data = cur_test[["pTB2"]].values
        cur_aux_data = cur_test[TrainingConfig.auxiliary_branches].values

        cut_2j = (cur_aux_data[:, TrainingConfig.auxiliary_branches.index("nJ")] == 2)
        cut_3j = (cur_aux_data[:, TrainingConfig.auxiliary_branches.index("nJ")] == 3)
        
        sig_aux_test.append(cur_aux_data)
        sig_aux_test_2j.append(cur_aux_data[cut_2j])
        sig_aux_test_3j.append(cur_aux_data[cut_3j])
        sig_data_test.append(cur_testdata)
        sig_data_test_2j.append(cur_testdata[cut_2j])
        sig_data_test_3j.append(cur_testdata[cut_3j])
        sig_mBB_test.append(cur_nuisdata)
        sig_mBB_test_2j.append(cur_nuisdata[cut_2j])
        sig_mBB_test_3j.append(cur_nuisdata[cut_3j])
        sig_weights_test.append(cur_weights)
        sig_weights_test_2j.append(cur_weights[cut_2j])
        sig_weights_test_3j.append(cur_weights[cut_3j])
        sig_dRBB_test.append(cur_dRBBdata)
        sig_dRBB_test_2j.append(cur_dRBBdata[cut_2j])
        sig_dRBB_test_3j.append(cur_dRBBdata[cut_3j])
        sig_pTB1_test.append(cur_pTB1data)
        sig_pTB2_test.append(cur_pTB2data)

    bkg_data_test = []
    bkg_data_test_2j = []
    bkg_data_test_3j = []
    bkg_aux_test = []
    bkg_aux_test_2j = []
    bkg_aux_test_3j = []
    bkg_mBB_test = []
    bkg_mBB_test_2j = []
    bkg_mBB_test_3j = []
    bkg_dRBB_test = []
    bkg_dRBB_test_2j = []
    bkg_dRBB_test_3j = []
    bkg_pTB1_test = []
    bkg_pTB2_test = []
    bkg_weights_test = []
    bkg_weights_test_2j = []
    bkg_weights_test_3j = []
    for sample, sample_name in zip(bkg_data, bkg_samples):
        cur_length = len(sample)
        sample = sample.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample
        cur_test = sample[int(data_slice[0] * cur_length) : int(data_slice[1] * cur_length)]
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights
        cur_dRBBdata = cur_test[["dRBB"]].values
        cur_pTB1data = cur_test[["pTB1"]].values
        cur_pTB2data = cur_test[["pTB2"]].values
        cur_aux_data = cur_test[TrainingConfig.auxiliary_branches].values

        cut_2j = cur_aux_data[:, TrainingConfig.auxiliary_branches.index("nJ")] == 2
        cut_3j = cur_aux_data[:, TrainingConfig.auxiliary_branches.index("nJ")] == 3

        bkg_aux_test.append(cur_aux_data)
        bkg_aux_test_2j.append(cur_aux_data[cut_2j])
        bkg_aux_test_3j.append(cur_aux_data[cut_3j])
        bkg_data_test.append(cur_testdata)
        bkg_data_test_2j.append(cur_testdata[cut_2j])
        bkg_data_test_3j.append(cur_testdata[cut_3j])
        bkg_mBB_test.append(cur_nuisdata)
        bkg_mBB_test_2j.append(cur_nuisdata[cut_2j])
        bkg_mBB_test_3j.append(cur_nuisdata[cut_3j])
        bkg_weights_test.append(cur_weights)
        bkg_weights_test_2j.append(cur_weights[cut_2j])
        bkg_weights_test_3j.append(cur_weights[cut_3j])
        bkg_dRBB_test.append(cur_dRBBdata)
        bkg_dRBB_test_2j.append(cur_dRBBdata[cut_2j])
        bkg_dRBB_test_3j.append(cur_dRBBdata[cut_3j])
        bkg_pTB1_test.append(cur_pTB1data)
        bkg_pTB2_test.append(cur_pTB2data)

    for model_dir in model_dirs:
        print("now evaluating " + model_dir)

        mce = AdversarialEnvironment.from_file(model_dir)
        plots_outdir = plot_dir

        # generate performance plots for each model individually
        ev = ModelEvaluator(mce)

        if plot_clf_distribs:
            # plot the output distribution of the classifier for a few events from each sample
            for sample, sample_name in zip(sig_data_test + bkg_data_test, sig_samples + bkg_samples):
                for event_num in range(10):
                    event = sample[[event_num]]
                    ev.plot_clf_pdf(event = event, varlabels = TrainingConfig.training_branches, plotlabel = sample_name, outpath = os.path.join(plots_outdir, "clf_pdf_" + sample_name + "_" + str(event_num) + ".pdf"))

        # generate plots showing the evolution of certain parameters during training
        tsp = TrainingStatisticsPlotter(model_dir)
        tsp.plot(outdir = plots_outdir)

        # plot the ROC curve as performance measure
        ev.plot_roc(data_sig = sig_data_test, data_bkg = bkg_data_test, aux_sig = sig_aux_test, aux_bkg = bkg_aux_test, sig_weights = sig_weights_test, bkg_weights = bkg_weights_test, outpath = plots_outdir)

        # generate distortion plots
        ev.plot_distortion(data_sig = sig_data_test_2j, data_bkg = bkg_data_test_2j, aux_sig = sig_aux_test_2j, aux_bkg = bkg_aux_test_2j,
                           var_sig = sig_mBB_test_2j, var_bkg = bkg_mBB_test_2j, 
                           weights_sig = sig_weights_test_2j, weights_bkg = bkg_weights_test_2j, sigeffs = [1.0, 0.5, 0.25], outpath = plots_outdir, 
                           labels_sig = sig_samples, labels_bkg = bkg_samples, xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.", path_prefix = "dist_mBB_2j")
        ev.plot_distortion(data_sig = sig_data_test_3j, data_bkg = bkg_data_test_3j, aux_sig = sig_aux_test_3j, aux_bkg = bkg_aux_test_3j,
                           var_sig = sig_mBB_test_3j, var_bkg = bkg_mBB_test_3j, 
                           weights_sig = sig_weights_test_3j, weights_bkg = bkg_weights_test_3j, sigeffs = [1.0, 0.5, 0.25], outpath = plots_outdir, 
                           labels_sig = sig_samples, labels_bkg = bkg_samples, xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.", path_prefix = "dist_mBB_3j")
        ev.plot_distortion(data_sig = sig_data_test_2j, data_bkg = bkg_data_test_2j, aux_sig = sig_aux_test_2j, aux_bkg = bkg_aux_test_2j,
                           var_sig = sig_dRBB_test_2j, var_bkg = bkg_dRBB_test_2j, 
                           weights_sig = sig_weights_test_2j, weights_bkg = bkg_weights_test_2j, sigeffs = [1.0, 0.5, 0.25], outpath = plots_outdir, 
                           labels_sig = sig_samples, labels_bkg = bkg_samples, xlabel = r'$\Delta R_{bb}$', ylabel = "a.u.", path_prefix = "dist_dRBB_2j", histrange = (0, 5))
        ev.plot_distortion(data_sig = sig_data_test_3j, data_bkg = bkg_data_test_3j, aux_sig = sig_aux_test_3j, aux_bkg = bkg_aux_test_3j,
                           var_sig = sig_dRBB_test_3j, var_bkg = bkg_dRBB_test_3j, 
                           weights_sig = sig_weights_test_3j, weights_bkg = bkg_weights_test_3j, sigeffs = [1.0, 0.5, 0.25], outpath = plots_outdir, 
                           labels_sig = sig_samples, labels_bkg = bkg_samples, xlabel = r'$\Delta R_{bb}$', ylabel = "a.u.", path_prefix = "dist_dRBB_3j", histrange = (0, 5))

        # ev.plot_distortion(data_sig = sig_data_test, data_bkg = bkg_data_test, var_sig = sig_pTB1_test, var_bkg = bkg_pTB1_test, 
        #                    weights_sig = sig_weights_test, weights_bkg = bkg_weights_test, sigeffs = [1.0, 0.5, 0.25], outpath = plots_outdir, 
        #                    labels_sig = sig_samples, labels_bkg = bkg_samples, xlabel = r'$p_{T, b(1)}$ [GeV]', ylabel = "a.u.", path_prefix = "dist_pTB1")
        # ev.plot_distortion(data_sig = sig_data_test, data_bkg = bkg_data_test, var_sig = sig_pTB2_test, var_bkg = bkg_pTB2_test, 
        #                    weights_sig = sig_weights_test, weights_bkg = bkg_weights_test, sigeffs = [1.0, 0.5, 0.25], outpath = plots_outdir, 
        #                    labels_sig = sig_samples, labels_bkg = bkg_samples, xlabel = r'$p_{T, b(2)}$ [GeV]', ylabel = "a.u.", path_prefix = "dist_pTB2")

        # plot classifier distributions
        #ev.plot_clf_distribution(data = sig_data_test + bkg_data_test, weights = sig_weights_test + bkg_weights_test, outpath = plots_outdir, labels = sig_samples + bkg_samples, num_cols = 2)

        # plot correlation plots of the classifier with m_BB
        #ev.plot_clf_correlations(varname = "mBB", data_sig = sig_data_test, weights_sig = sig_weights_test, labels_sig = sig_samples, data_bkg = bkg_data_test, weights_bkg = bkg_weights_test, labels_bkg = bkg_samples, outpath = plots_outdir)

        # # get inclusive performance metrics and save them
        # perfdict = ev.get_performance_metrics(sig_data_test, bkg_data_test, sig_mBB_test, bkg_mBB_test, sig_weights_test, 
        #                                       bkg_weights_test, labels_sig = sig_samples, labels_bkg = bkg_samples)
        # print("got inclusive perfdict = " + str(perfdict))

        # # get performance metrics for the SR mass range only (from 30 GeV < mBB < 210 GeV)
        # sig_data_test_SR = []
        # bkg_data_test_SR = []
        # sig_mBB_test_SR = []
        # bkg_mBB_test_SR = []
        # sig_weights_test_SR = []
        # bkg_weights_test_SR = []

        # for cur_data, cur_mBB, cur_weights in zip(sig_data_test, sig_mBB_test, sig_weights_test):
        #     cut_pass = np.logical_and.reduce((cur_mBB > 30, cur_mBB < 210)).flatten()
        #     sig_data_test_SR.append(cur_data[cut_pass])
        #     sig_mBB_test_SR.append(cur_mBB[cut_pass])
        #     sig_weights_test_SR.append(cur_weights[cut_pass])

        # for cur_data, cur_mBB, cur_weights in zip(bkg_data_test, bkg_mBB_test, bkg_weights_test):
        #     cut_pass = np.logical_and.reduce((cur_mBB > 30, cur_mBB < 210)).flatten()
        #     bkg_data_test_SR.append(cur_data[cut_pass])
        #     bkg_mBB_test_SR.append(cur_mBB[cut_pass])
        #     bkg_weights_test_SR.append(cur_weights[cut_pass])

        # perfdict_SR = ev.get_performance_metrics(sig_data_test_SR, bkg_data_test_SR, sig_mBB_test_SR, bkg_mBB_test_SR, sig_weights_test_SR, 
        #                                          bkg_weights_test_SR, labels_sig = sig_samples, labels_bkg = bkg_samples, prefix = "SR_")
        # print("got SR perfdict = " + str(perfdict_SR))

        # perfdict.update(perfdict_SR)

        # # save the combined perfdict
        # with open(os.path.join(plots_outdir, "perfdict.pkl"), "wb") as outfile:
        #    pickle.dump(perfdict, outfile)

if __name__ == "__main__":
    main()
