import os, pickle
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.AdversarialEnvironment import AdversarialEnvironment
from analysis.Category import Category
from analysis.CutBasedCategoryFiller import CutBasedCategoryFiller
from analysis.ClassifierBasedCategoryFiller import ClassifierBasedCategoryFiller
from plotting.CategoryPlotter import CategoryPlotter
from plotting.ModelEvaluator import ModelEvaluator
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
        inclusive.add_events(events = events, weights = weights, process = process, event_variables = TrainingConfig.training_branches)

    # prepare the common binning for all signal regions (all values in GeV)
    SR_low = 30
    SR_up = 210
    SR_binwidth = 10
    SR_mBB_binning = np.linspace(SR_low, SR_up, num = int((SR_up - SR_low) / SR_binwidth), endpoint = True)
    dRBB_binning = np.linspace(0, 5, num = 25, endpoint = True)

    # prepare the inclusive binning
    inclusive_low = 30
    inclusive_high = 600
    inclusive_binwidth = 10
    inclusive_binning = np.linspace(inclusive_low, inclusive_high, num = int((inclusive_high - inclusive_low) / inclusive_binwidth), endpoint = True)

    # prepare the dictionary holding all the sensitivity values
    sensdict = {}

    # show the inclusive event content
    CategoryPlotter.plot_category_composition(inclusive, binning = inclusive_binning, outpath = os.path.join(plotdir, "dist_mBB_inclusive.pdf"), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', plotlabel = ["MC16d", "inclusive"])

    inclusive_events = inclusive.get_number_events("Hbb")
    print("have {} signal events in total".format(inclusive_events))

    # export the inclusive background distribution
    inclusive.export_histogram(binning = SR_mBB_binning, processes = bkg_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_inclusive.pkl"), clipping = False)    
    inclusive.export_histogram(binning = dRBB_binning, processes = bkg_samples, var_name = "dRBB", outfile = os.path.join(plotdir, "dist_dRBB_inclusive.pkl"), clipping = False)

    # measure its binned significance
    significance_inclusive = inclusive.get_binned_significance(binning = inclusive_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")
    sensdict["significance_inclusive"] = significance_inclusive

    # also fill inclusive 2- and 3-jet categories to get a baseline for the shapes
    inclusive_2J = CutBasedCategoryFiller.create_nJ_category(process_events = data_test,
                                                             process_aux_events = aux_test,
                                                             process_weights = weights_test,
                                                             process_names = samples,
                                                             nJ = 2)

    inclusive_3J = CutBasedCategoryFiller.create_nJ_category(process_events = data_test,
                                                             process_aux_events = aux_test,
                                                             process_weights = weights_test,
                                                             process_names = samples,
                                                             nJ = 3)

    # and export their combined background distribution
    inclusive_2J.export_histogram(binning = SR_mBB_binning, processes = bkg_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_inclusive_2J.pkl"), clipping = False)    
    inclusive_2J.export_histogram(binning = SR_mBB_binning, processes = sig_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_sig_inclusive_2J.pkl"), clipping = False)    
    inclusive_2J.export_histogram(binning = dRBB_binning, processes = bkg_samples, var_name = "dRBB", outfile = os.path.join(plotdir, "dist_dRBB_inclusive_2J.pkl"), clipping = False)

    inclusive_3J.export_histogram(binning = SR_mBB_binning, processes = bkg_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_inclusive_3J.pkl"), clipping = False)    
    inclusive_3J.export_histogram(binning = SR_mBB_binning, processes = sig_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_sig_inclusive_3J.pkl"), clipping = False)    
    inclusive_3J.export_histogram(binning = dRBB_binning, processes = bkg_samples, var_name = "dRBB", outfile = os.path.join(plotdir, "dist_dRBB_inclusive_3J.pkl"), clipping = False)

    # create the event categories for the simple cut-based analysis
    for cur_nJ, cur_inclusive_cat in zip([2, 3], [inclusive_2J, inclusive_3J]):
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

        # compute and store the signal efficiencies of these categories
        sensdict["sigeff_low_MET_{}J".format(cur_nJ)] = low_MET_cat.get_number_events("Hbb") / inclusive_events
        sensdict["sigeff_high_MET_{}J".format(cur_nJ)] = high_MET_cat.get_number_events("Hbb") / inclusive_events

        # compute their expected sensitivities
        significance_low_MET = low_MET_cat.get_binned_significance(binning = SR_mBB_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")
        significance_high_MET = high_MET_cat.get_binned_significance(binning = SR_mBB_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")

        sensdict["significance_low_MET_{}J".format(cur_nJ)] = significance_low_MET
        sensdict["significance_high_MET_{}J".format(cur_nJ)] = significance_high_MET

        # ----------------------------------------------------------------------------------------
        # compute the distortions to m_BB in the combined background caused by these categories
        # ----------------------------------------------------------------------------------------
        mBB_inclusive, mBB_inclusive_weights = cur_inclusive_cat.get_event_variable(processes = bkg_samples, var = "mBB")
        mBB_cat, mBB_cat_weights = low_MET_cat.get_event_variable(processes = bkg_samples, var = "mBB")
        
        # do it both for the inclusive mass range (i.e. get a "global" KS value) ...
        cut = np.logical_and.reduce((mBB_inclusive > 30, mBB_inclusive < 210)).flatten()
        mBB_inclusive_red, mBB_inclusive_red_weights = mBB_inclusive[cut], mBB_inclusive_weights[cut]

        # ... and also for the range around the Higgs peak only (i.e. 30 - 210 GeV)
        cut = np.logical_and.reduce((mBB_cat > 30, mBB_cat < 210)).flatten()
        mBB_cat_red, mBB_cat_red_weights = mBB_cat[cut], mBB_cat_weights[cut]

        KS_low_MET_cat = ModelEvaluator._get_KS(mBB_inclusive, mBB_inclusive_weights, mBB_cat, mBB_cat_weights, cums_outfile = os.path.join(plotdir, "low_MET_KS_cums_{}J.pdf".format(cur_nJ)))
        KS_low_MET_cat_red = ModelEvaluator._get_KS(mBB_inclusive_red, mBB_inclusive_red_weights, mBB_cat_red, mBB_cat_red_weights, cums_outfile = os.path.join(plotdir, "low_MET_KS_red_cums_{}J.pdf".format(cur_nJ)))
        sensdict["KS_bkg_low_MET_{}J".format(cur_nJ)] = KS_low_MET_cat
        sensdict["KS_bkg_low_MET_{}J_red".format(cur_nJ)] = KS_low_MET_cat_red

        mBB_inclusive, mBB_inclusive_weights = cur_inclusive_cat.get_event_variable(processes = bkg_samples, var = "mBB")
        mBB_cat, mBB_cat_weights = high_MET_cat.get_event_variable(processes = bkg_samples, var = "mBB")

        # do it both for the inclusive mass range (i.e. get a "global" KS value) ...
        cut = np.logical_and.reduce((mBB_inclusive > 30, mBB_inclusive < 210)).flatten()
        mBB_inclusive_red, mBB_inclusive_red_weights = mBB_inclusive[cut], mBB_inclusive_weights[cut]

        # ... and also for the range around the Higgs peak only (i.e. 30 - 210 GeV)
        cut = np.logical_and.reduce((mBB_cat > 30, mBB_cat < 210)).flatten()
        mBB_cat_red, mBB_cat_red_weights = mBB_cat[cut], mBB_cat_weights[cut]        

        KS_high_MET_cat = ModelEvaluator._get_KS(mBB_inclusive, mBB_inclusive_weights, mBB_cat, mBB_cat_weights, cums_outfile = os.path.join(plotdir, "high_MET_KS_cums_{}J.pdf".format(cur_nJ)))
        KS_high_MET_cat_red = ModelEvaluator._get_KS(mBB_inclusive_red, mBB_inclusive_red_weights, mBB_cat_red, mBB_cat_red_weights, cums_outfile = os.path.join(plotdir, "high_MET_KS_red_cums_{}J.pdf".format(cur_nJ)))
        sensdict["KS_bkg_high_MET_{}J".format(cur_nJ)] = KS_high_MET_cat
        sensdict["KS_bkg_high_MET_{}J_red".format(cur_nJ)] = KS_high_MET_cat_red

        # also show the distributions in these two categories
        CategoryPlotter.plot_category_composition(low_MET_cat, binning = SR_mBB_binning, outpath = os.path.join(plotdir, "dist_mBB_low_MET_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", r'150 GeV < MET < 200 GeV', "dRBB < 1.8", "nJ = {}".format(cur_nJ)], args = {})
        CategoryPlotter.plot_category_composition(high_MET_cat, binning = SR_mBB_binning, outpath = os.path.join(plotdir, "dist_mBB_high_MET_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", "MET > 200 GeV", "dRBB < 1.2", "nJ = {}".format(cur_nJ)], args = {})

        # save the distribution of the combined background ...
        low_MET_cat.export_histogram(binning = SR_mBB_binning, processes = bkg_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_low_MET_{}J.pkl".format(cur_nJ)), clipping = False)
        high_MET_cat.export_histogram(binning = SR_mBB_binning, processes = bkg_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_high_MET_{}J.pkl".format(cur_nJ)), clipping = False)

        # ... and the signal
        low_MET_cat.export_histogram(binning = SR_mBB_binning, processes = sig_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_sig_low_MET_{}J.pkl".format(cur_nJ)), clipping = False)
        high_MET_cat.export_histogram(binning = SR_mBB_binning, processes = sig_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_sig_high_MET_{}J.pkl".format(cur_nJ)), clipping = False)

        low_MET_cat.export_histogram(binning = dRBB_binning, processes = bkg_samples, var_name = "dRBB", outfile = os.path.join(plotdir, "dist_dRBB_low_MET_{}J.pkl".format(cur_nJ)), clipping = False)
        high_MET_cat.export_histogram(binning = dRBB_binning, processes = bkg_samples, var_name = "dRBB", outfile = os.path.join(plotdir, "dist_dRBB_high_MET_{}J.pkl".format(cur_nJ)), clipping = False)

    # load the classifier model and also fill two classifier-based categories
    env = AdversarialEnvironment.from_file(model_dir)

    for cur_nJ, cur_inclusive_cat in zip([2, 3], [inclusive_2J, inclusive_3J]):
        class_cat_tight = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                   process_events = data_test,
                                                                                   process_aux_events = aux_test,
                                                                                   process_weights = weights_test,
                                                                                   process_names = samples,
                                                                                   signal_events = sig_data_test,
                                                                                   signal_weights = sig_weights_test,
                                                                                   classifier_sigeff_range = (0.30, 0.0),
                                                                                   nJ = cur_nJ)

        class_cat_loose = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                   process_events = data_test,
                                                                                   process_aux_events = aux_test,
                                                                                   process_weights = weights_test,
                                                                                   process_names = samples,
                                                                                   signal_events = sig_data_test,
                                                                                   signal_weights = sig_weights_test,
                                                                                   classifier_sigeff_range = (0.80, 0.30),
                                                                                   nJ = cur_nJ)        

        # compute and store the signal efficiencies of these categories
        sensdict["sigeff_clf_tight_{}J".format(cur_nJ)] = class_cat_tight.get_number_events("Hbb") / inclusive_events
        sensdict["sigeff_clf_loose_{}J".format(cur_nJ)] = class_cat_loose.get_number_events("Hbb") / inclusive_events

        # compute the expected sensitivities
        significance_clf_loose = class_cat_loose.get_binned_significance(binning = SR_mBB_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")
        significance_clf_tight = class_cat_tight.get_binned_significance(binning = SR_mBB_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")

        sensdict["significance_clf_loose_{}J".format(cur_nJ)] = significance_clf_loose
        sensdict["significance_clf_tight_{}J".format(cur_nJ)] = significance_clf_tight

        # compute the distortions to m_BB caused by these categories
        mBB_inclusive, mBB_inclusive_weights = cur_inclusive_cat.get_event_variable(processes = bkg_samples, var = "mBB")
        mBB_cat, mBB_cat_weights = class_cat_tight.get_event_variable(processes = bkg_samples, var = "mBB")

        # do it both for the inclusive mass range (i.e. get a "global" KS value) ...
        cut = np.logical_and.reduce((mBB_inclusive > 30, mBB_inclusive < 210)).flatten()
        mBB_inclusive_red, mBB_inclusive_red_weights = mBB_inclusive[cut], mBB_inclusive_weights[cut]

        # ... and also for the range around the Higgs peak only (i.e. 30 - 210 GeV)
        cut = np.logical_and.reduce((mBB_cat > 30, mBB_cat < 210)).flatten()
        mBB_cat_red, mBB_cat_red_weights = mBB_cat[cut], mBB_cat_weights[cut]

        KS_class_cat_tight = ModelEvaluator._get_KS(mBB_inclusive, mBB_inclusive_weights, mBB_cat, mBB_cat_weights, cums_outfile = os.path.join(plotdir, "class_tight_KS_cums_{}J.pdf".format(cur_nJ)))
        KS_class_cat_tight_red = ModelEvaluator._get_KS(mBB_inclusive_red, mBB_inclusive_red_weights, mBB_cat_red, mBB_cat_red_weights, cums_outfile = os.path.join(plotdir, "class_tight_KS_red_cums_{}J.pdf".format(cur_nJ)))
        sensdict["KS_bkg_class_tight_{}J".format(cur_nJ)] = KS_class_cat_tight
        sensdict["KS_bkg_class_tight_{}J_red".format(cur_nJ)] = KS_class_cat_tight_red

        mBB_inclusive, mBB_inclusive_weights = cur_inclusive_cat.get_event_variable(processes = bkg_samples, var = "mBB")
        mBB_cat, mBB_cat_weights = class_cat_loose.get_event_variable(processes = bkg_samples, var = "mBB")

        # do it both for the inclusive mass range (i.e. get a "global" KS value) ...
        cut = np.logical_and.reduce((mBB_inclusive > 30, mBB_inclusive < 210)).flatten()
        mBB_inclusive_red, mBB_inclusive_red_weights = mBB_inclusive[cut], mBB_inclusive_weights[cut]

        # ... and also for the range around the Higgs peak only (i.e. 30 - 210 GeV)
        cut = np.logical_and.reduce((mBB_cat > 30, mBB_cat < 210)).flatten()
        mBB_cat_red, mBB_cat_red_weights = mBB_cat[cut], mBB_cat_weights[cut]

        KS_class_cat_loose = ModelEvaluator._get_KS(mBB_inclusive, mBB_inclusive_weights, mBB_cat, mBB_cat_weights, cums_outfile = os.path.join(plotdir, "class_loose_KS_cums_{}J.pdf".format(cur_nJ)))
        KS_class_cat_loose_red = ModelEvaluator._get_KS(mBB_inclusive_red, mBB_inclusive_red_weights, mBB_cat_red, mBB_cat_red_weights, cums_outfile = os.path.join(plotdir, "class_loose_KS_red_cums_{}J.pdf".format(cur_nJ)))
        sensdict["KS_bkg_class_loose_{}J".format(cur_nJ)] = KS_class_cat_loose
        sensdict["KS_bkg_class_loose_{}J_red".format(cur_nJ)] = KS_class_cat_loose_red

        # plot their signal composition
        CategoryPlotter.plot_category_composition(class_cat_tight, binning = SR_mBB_binning, outpath = os.path.join(plotdir, "dist_mBB_class_tight_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", "clf tight", "nJ = {}".format(cur_nJ)])
        CategoryPlotter.plot_category_composition(class_cat_loose, binning = SR_mBB_binning, outpath = os.path.join(plotdir, "dist_mBB_class_loose_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", "clf loose", "nJ = {}".format(cur_nJ)])

        # save the distribution of the combined background ...
        class_cat_tight.export_histogram(binning = SR_mBB_binning, processes = bkg_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_class_tight_{}J.pkl".format(cur_nJ)), clipping = False)
        class_cat_loose.export_histogram(binning = SR_mBB_binning, processes = bkg_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_class_loose_{}J.pkl".format(cur_nJ)), clipping = False)

        # ... and the signal
        class_cat_tight.export_histogram(binning = SR_mBB_binning, processes = sig_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_sig_class_tight_{}J.pkl".format(cur_nJ)), clipping = False)
        class_cat_loose.export_histogram(binning = SR_mBB_binning, processes = sig_samples, var_name = "mBB", outfile = os.path.join(plotdir, "dist_mBB_sig_class_loose_{}J.pkl".format(cur_nJ)), clipping = False)

        class_cat_tight.export_histogram(binning = dRBB_binning, processes = bkg_samples, var_name = "dRBB", outfile = os.path.join(plotdir, "dist_dRBB_class_tight_{}J.pkl".format(cur_nJ)), clipping = False)
        class_cat_loose.export_histogram(binning = dRBB_binning, processes = bkg_samples, var_name = "dRBB", outfile = os.path.join(plotdir, "dist_dRBB_class_loose_{}J.pkl".format(cur_nJ)), clipping = False)

    # plot the summarized significance values and save them
    sensdict.update(env.create_paramdict())
    print("got sensdict = " + str(sensdict))
    with open(os.path.join(plotdir, "sensdict.pkl"), "wb") as outfile:
        pickle.dump(sensdict, outfile)

if __name__ == "__main__":
    main()
