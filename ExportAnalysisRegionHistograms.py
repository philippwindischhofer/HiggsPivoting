import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models.AdversarialEnvironment import AdversarialEnvironment
from analysis.Category import Category
from analysis.ClassifierBasedCategoryFiller import ClassifierBasedCategoryFiller
from analysis.CutBasedCategoryFiller import CutBasedCategoryFiller
from base.Configs import TrainingConfig
from plotting.CategoryPlotter import CategoryPlotter
from DatasetExtractor import TrainNuisAuxSplit

def main():
    parser = ArgumentParser(description = "populate analysis signal regions and export them to be used with HistFitter")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--model_dir", action = "store", dest = "model_dir")
    parser.add_argument("--test_size", action = "store", dest = "test_size")
    parser.add_argument("--out_dir", action = "store", dest = "out_dir")
    args = vars(parser.parse_args())

    infile_path = args["infile_path"]
    model_dir = args["model_dir"]
    outdir = args["out_dir"]
    test_size = float(args["test_size"]) # need this when reading datasets that haven't been used for training (but instead for assessing systematics)

    sig_samples = ["Hbb"]
    bkg_samples = ["ttbar", "Zjets", "Wjets", "diboson", "singletop"]

    data_sig = [pd.read_hdf(infile_path, key = sample) for sample in sig_samples]
    data_bkg = [pd.read_hdf(infile_path, key = sample) for sample in bkg_samples]

    # load all signal processes
    sig_data_test = [] # this holds all the branches used as inputs to the classifier
    sig_weights_test = []
    sig_aux_data_test = [] # this holds some other branches that may be important
    for sample in data_sig:
        _, cur_test = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights

        cur_aux_data = cur_test[TrainingConfig.other_branches].values
        sig_data_test.append(cur_testdata)
        sig_weights_test.append(cur_weights / test_size)
        sig_aux_data_test.append(cur_aux_data)

    # load all background processes
    bkg_data_test = [] # this holds all the branches used as inputs to the classifier
    bkg_weights_test = []
    bkg_aux_data_test = [] # this holds some other branches that may be important
    for sample in data_bkg:
        _, cur_test = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights

        cur_aux_data = cur_test[TrainingConfig.other_branches].values
        bkg_data_test.append(cur_testdata)
        bkg_weights_test.append(cur_weights / test_size)
        bkg_aux_data_test.append(cur_aux_data)

    # also prepare the total, concatenated versions
    data_test = sig_data_test + bkg_data_test
    aux_test = sig_aux_data_test + bkg_aux_data_test
    weights_test = sig_weights_test + bkg_weights_test
    samples = sig_samples + bkg_samples

    # load the AdversarialEnvironment
    env = AdversarialEnvironment.from_file(model_dir)

    # prepare the common mBB binning for all signal regions
    SR_low = 30
    SR_up = 210
    SR_binwidth = 10
    SR_binning = np.linspace(SR_low, SR_up, num = int((SR_up - SR_low) / SR_binwidth), endpoint = True)

    # also prepare the binning along the MVA dimension
    sigeff_low = 0
    sigeff_up = 1
    sigeff_binwidth = 0.05 # make bins, each of which has a signal efficiency of 5%
    sigeff_binning = np.linspace(sigeff_low, sigeff_up, num = int((sigeff_up - sigeff_low) / sigeff_binwidth), endpoint = True)

    # now convert the binning in terms of signal efficiency into the actual binning
    # in terms of the classifier output value
    MVA_binning = [ClassifierBasedCategoryFiller._sigeff_to_score(env = env, signal_events = sig_data_test, signal_weights = sig_weights_test, sigeff = sigeff) for sigeff in sigeff_binning[::-1]]

    print("signal efficiency binning: {}".format(sigeff_binning))
    print("classifier output binning: {}".format(MVA_binning))

    for cur_nJ in [2, 3]:
        # first, export the categories of the cut-based analysis: high / low MET
        low_MET_cat = CutBasedCategoryFiller.create_low_MET_category(process_events = data_test,
                                                                     process_aux_events = aux_test,
                                                                     process_weights = weights_test,
                                                                     process_names = samples,
                                                                     nJ = cur_nJ)

        low_MET_cat.export_ROOT_histogram(binning = SR_binning, processes = sig_samples + bkg_samples, var_names = "mBB",
                                          outfile_path = os.path.join(outdir, "{}jet_low_MET.root".format(cur_nJ)), clipping = False, density = False)

        CategoryPlotter.plot_category_composition(low_MET_cat, binning = SR_binning, outpath = os.path.join(outdir, "{}jet_low_MET.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", r'150 GeV < MET < 200 GeV', "dRBB < 1.8", "nJ = {}".format(cur_nJ)], args = {})

        high_MET_cat = CutBasedCategoryFiller.create_high_MET_category(process_events = data_test,
                                                                       process_aux_events = aux_test,
                                                                       process_weights = weights_test,
                                                                       process_names = samples,
                                                                       nJ = cur_nJ)

        high_MET_cat.export_ROOT_histogram(binning = SR_binning, processes = sig_samples + bkg_samples, var_names = "mBB",
                                          outfile_path = os.path.join(outdir, "{}jet_high_MET.root".format(cur_nJ)), clipping = False, density = False)

        CategoryPlotter.plot_category_composition(high_MET_cat, binning = SR_binning, outpath = os.path.join(outdir, "{}jet_high_MET.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", "MET > 200 GeV", "dRBB < 1.2", "nJ = {}".format(cur_nJ)], args = {})

        # prepare and export three SR/CRs per jet category:
        # a very tight analysis category, highly enriched in signal ...
        class_cat_tight = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                   process_events = data_test,
                                                                                   process_aux_events = aux_test,
                                                                                   process_weights = weights_test,
                                                                                   process_names = samples,
                                                                                   signal_events = sig_data_test,
                                                                                   signal_weights = sig_weights_test,
                                                                                   classifier_sigeff_range = (0.30, 0.0),
                                                                                   nJ = cur_nJ)

        class_cat_tight.export_ROOT_histogram(binning = SR_binning, processes = sig_samples + bkg_samples, var_names = "mBB", 
                                              outfile_path = os.path.join(outdir, "{}jet_tight.root".format(cur_nJ)), clipping = False, density = False)

        CategoryPlotter.plot_category_composition(class_cat_tight, binning = SR_binning, outpath = os.path.join(outdir, "dist_mBB_class_tight_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", "clf tight", "nJ = {}".format(cur_nJ)])

        # ... a loose category with high event yield but low signal purity ...
        class_cat_loose = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                   process_events = data_test,
                                                                                   process_aux_events = aux_test,
                                                                                   process_weights = weights_test,
                                                                                   process_names = samples,
                                                                                   signal_events = sig_data_test,
                                                                                   signal_weights = sig_weights_test,
                                                                                   classifier_sigeff_range = (0.80, 0.30),
                                                                                   nJ = cur_nJ)

        CategoryPlotter.plot_category_composition(class_cat_loose, binning = SR_binning, outpath = os.path.join(outdir, "dist_mBB_class_loose_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", "clf loose", "nJ = {}".format(cur_nJ)])

        class_cat_loose.export_ROOT_histogram(binning = SR_binning, processes = sig_samples + bkg_samples, var_names = "mBB", 
                                              outfile_path = os.path.join(outdir, "{}jet_loose.root".format(cur_nJ)), clipping = False, density = False)

        # ... and also a signal-depleted region that constrains the backgrounds
        class_cat_depleted = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                process_events = data_test,
                                                                                process_aux_events = aux_test,
                                                                                process_weights = weights_test,
                                                                                process_names = samples,
                                                                                signal_events = sig_data_test,
                                                                                signal_weights = sig_weights_test,
                                                                                classifier_sigeff_range = (1.00, 0.80),
                                                                                nJ = cur_nJ)

        class_cat_depleted.export_ROOT_histogram(binning = SR_binning, processes = sig_samples + bkg_samples, var_names = "mBB", 
                                                 outfile_path = os.path.join(outdir, "{}jet_depleted.root".format(cur_nJ)), clipping = False, density = False)

        CategoryPlotter.plot_category_composition(class_cat_depleted, binning = SR_binning, outpath = os.path.join(outdir, "dist_mBB_class_depleted_{}J.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", "clf loose", "nJ = {}".format(cur_nJ)])

        # now, also export the classifier categories for each jet split
        class_cat_inclusive = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                       process_events = data_test,
                                                                                       process_aux_events = aux_test,
                                                                                       process_weights = weights_test,
                                                                                       process_names = samples,
                                                                                       signal_events = sig_data_test,
                                                                                       signal_weights = sig_weights_test,
                                                                                       classifier_sigeff_range = (1.0, 0.0),
                                                                                       nJ = cur_nJ)

        class_cat_inclusive.export_ROOT_histogram(binning = MVA_binning, processes = sig_samples + bkg_samples, var_names = "clf", 
                                                  outfile_path = os.path.join(outdir, "{}jet_MVA.root".format(cur_nJ)), clipping = False, density = False, ignore_binning = True)

        CategoryPlotter.plot_category_composition(class_cat_inclusive, binning = MVA_binning, outpath = os.path.join(outdir, "dist_MVA_{}J.pdf".format(cur_nJ)), var = "clf", xlabel = r'MVA', 
                                                  plotlabel = ["MC16d", "MVA", "nJ = {}".format(cur_nJ)], logscale = True)
        
if __name__ == "__main__":
    main()
