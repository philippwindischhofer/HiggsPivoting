import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.AdversarialEnvironment import AdversarialEnvironment
from analysis.Category import Category
from analysis.CutBasedCategoryFiller import CutBasedCategoryFiller
from analysis.ClassifierBasedCategoryFiller import ClassifierBasedCategoryFiller
from plotting.CategoryPlotter import CategoryPlotter
from base.Configs import TrainingConfig
from DatasetExtractor import TrainNuisAuxSplit

def main():
    parser = ArgumentParser(description = "evaluate expected sensitivity")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--model_dir", action = "store", dest = "model_dir")
    parser.add_argument("--plot_dir", action = "store", dest = "plot_dir")
    args = vars(parser.parse_args())

    infile_path = args["infile_path"]
    model_dir = args["model_dir"]
    plotdir = args["plot_dir"]

    test_size = TrainingConfig.test_size # fraction of MC16d events used for the estimation of the expected sensitivity (therefore need to scale up the results by the inverse of this factor)

    # read the test dataset, which will be used to get the expected sensitivity of the analysis
    sig_samples = ["Hbb"]
    bkg_samples = ["ttbar", "Zjets", "Wjets", "diboson", "singletop"]

    print("loading data ...")
    sig_data = [pd.read_hdf(infile_path, key = sig_sample) for sig_sample in sig_samples]
    bkg_data = [pd.read_hdf(infile_path, key = bkg_sample) for bkg_sample in bkg_samples]

    sig_data_test = []
    sig_mBB_test = []
    sig_weights_test = []
    sig_aux_data_test = []
    for sample in sig_data:
        _, cur_test = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights
        cur_aux_data = cur_test[TrainingConfig.other_branches].values
        sig_data_test.append(cur_testdata)
        sig_mBB_test.append(cur_nuisdata)
        sig_weights_test.append(cur_weights / test_size)
        sig_aux_data_test.append(cur_aux_data)

    bkg_data_test = []
    bkg_mBB_test = []
    bkg_weights_test = []
    bkg_aux_data_test = []
    for sample in bkg_data:
        _, cur_test = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights
        cur_aux_data = cur_test[TrainingConfig.other_branches].values
        bkg_data_test.append(cur_testdata)
        bkg_mBB_test.append(cur_nuisdata)
        bkg_weights_test.append(cur_weights / test_size)
        bkg_aux_data_test.append(cur_aux_data)

    # also prepare the total, concatenated versions
    data_test = sig_data_test + bkg_data_test
    aux_test = sig_aux_data_test + bkg_aux_data_test
    weights_test = sig_weights_test + bkg_weights_test
    samples = sig_samples + bkg_samples

    # first, plot the total event content (i.e. corresponding to an "inclusive" event category)
    inclusive = Category("inclusive")
    for events, weights, process in zip(sig_data_test + bkg_data_test, sig_weights_test + bkg_weights_test, sig_samples + bkg_samples):
        inclusive.add_events(events = events, weights = weights, process = process)

    # prepare the common binning for all signal regions (all values in GeV)
    SR_low = 30
    SR_up = 210
    SR_binwidth = 10
    SR_binning = np.linspace(SR_low, SR_up, num = int((SR_up - SR_low) / SR_binwidth), endpoint = True)

    # prepare the inclusive binning
    inclusive_low = 30
    inclusive_high = 600
    inclusive_binwidth = 10
    inclusive_binning = np.linspace(inclusive_low, inclusive_high, num = int((inclusive_high - inclusive_low) / inclusive_binwidth), endpoint = True)

    # prepare the dictionary holding all the sensitivity values
    sensdict = {}

    # show the inclusive event content
    CategoryPlotter.plot_category_composition(inclusive, binning = inclusive_binning, outpath = os.path.join(plotdir, "dist_mBB_inclusive.pdf"), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', plotlabel = ["MC16d only", "inclusive"])

    # measure its binned significance
    significance_inclusive = inclusive.get_binned_significance(binning = inclusive_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")
    sensdict["significance_inclusive"] = significance_inclusive

    # create the event categories for the simple cut-based analysis
    for cur_nJ in [2, 3]:
        low_MET_cat = CutBasedCategoryFiller.create_low_MET_category(process_events = data_test,
                                                                     process_aux_events = aux_test,
                                                                     process_weights = weights_test,
                                                                     process_names = samples,
                                                                     nJ = cur_nJ)
        high_MET_cat = CutBasedCategoryFiller.create_high_MET_category(process_events = data_test,
                                                                       process_aux_events = aux_test,
                                                                       process_weights = weights_test,
                                                                       process_names = samples,
                                                                       nJ = cur_nJ)

        significance_low_MET = low_MET_cat.get_binned_significance(binning = SR_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")
        significance_high_MET = high_MET_cat.get_binned_significance(binning = SR_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")

        sensdict["significance_low_MET_{}J".format(cur_nJ)] = significance_low_MET
        sensdict["significance_high_MET_{}J".format(cur_nJ)] = significance_high_MET

        # also show the distributions in these two categories
        CategoryPlotter.plot_category_composition(low_MET_cat, binning = SR_binning, outpath = os.path.join(plotdir, "dist_mBB_low_MET_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d only", r'150 GeV < MET < 200 GeV', "dRBB < 1.8", "nJ = {}".format(cur_nJ)], args = {"range": (25, 200)})
        CategoryPlotter.plot_category_composition(high_MET_cat, binning = SR_binning, outpath = os.path.join(plotdir, "dist_mBB_high_MET_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d only", "MET > 200 GeV", "dRBB < 1.2", "nJ = {}".format(cur_nJ)], args = {"range": (25, 200)})

    # load the classifier model and also fill two classifier-based categories
    env = AdversarialEnvironment.from_file(model_dir)

    for cur_nJ in [2, 3]:
        class_cat_tight = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                   process_events = data_test,
                                                                                   process_aux_events = aux_test,
                                                                                   process_weights = weights_test,
                                                                                   process_names = samples,
                                                                                   signal_events = sig_data_test,
                                                                                   signal_weights = sig_weights_test,
                                                                                   classifier_sigeff_range = (0.25, 0.0),
                                                                                   nJ = cur_nJ)

        class_cat_loose = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                   process_events = data_test,
                                                                                   process_aux_events = aux_test,
                                                                                   process_weights = weights_test,
                                                                                   process_names = samples,
                                                                                   signal_events = sig_data_test,
                                                                                   signal_weights = sig_weights_test,
                                                                                   classifier_sigeff_range = (0.75, 0.25),
                                                                                   nJ = cur_nJ)        

        significance_clf_loose = class_cat_loose.get_binned_significance(binning = SR_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")
        significance_clf_tight = class_cat_tight.get_binned_significance(binning = SR_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")

        sensdict["significance_clf_loose_{}J".format(cur_nJ)] = significance_clf_loose
        sensdict["significance_clf_tight_{}J".format(cur_nJ)] = significance_clf_tight

        CategoryPlotter.plot_category_composition(class_cat_tight, binning = SR_binning, outpath = os.path.join(plotdir, "dist_mBB_class_tight_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d only", "clf tight", "nJ = {}".format(cur_nJ)], args = {"range": (25, 200)})
        CategoryPlotter.plot_category_composition(class_cat_loose, binning = SR_binning, outpath = os.path.join(plotdir, "dist_mBB_class_loose_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d only", "clf loose", "nJ = {}".format(cur_nJ)], args = {"range": (25, 200)})


    # plot the summarized significance values
    print(sensdict)

if __name__ == "__main__":
    main()
