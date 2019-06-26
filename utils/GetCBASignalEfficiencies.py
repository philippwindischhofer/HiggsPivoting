import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from base.Configs import TrainingConfig
from analysis.CutBasedCategoryFiller import CutBasedCategoryFiller
from plotting.CategoryPlotter import CategoryPlotter
from DatasetExtractor import TrainNuisAuxSplit
from base.Configs import TrainingConfig

def GetCBASignalEfficiencies(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    sig_samples = TrainingConfig.sig_samples
    bkg_samples = TrainingConfig.bkg_samples
    infile_path = TrainingConfig.data_path

    data_slice = TrainingConfig.validation_slice
    slice_size = data_slice[1] - data_slice[0]

    data_sig = [pd.read_hdf(infile_path, key = sample) for sample in sig_samples]
    data_bkg = [pd.read_hdf(infile_path, key = sample) for sample in bkg_samples]

    # load all signal processes
    sig_data_test = [] # this holds all the branches used as inputs to the classifier
    sig_weights_test = []
    sig_aux_data_test = [] # this holds some other branches that may be important
    for sample, sample_name in zip(data_sig, sig_samples):
        cur_length = len(sample)
        sample = sample.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample
        cur_test = sample[int(data_slice[0] * cur_length) : int(data_slice[1] * cur_length)]
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights

        cur_aux_data = cur_test[TrainingConfig.auxiliary_branches].values
        sig_data_test.append(cur_testdata)
        sig_weights_test.append(cur_weights / slice_size * TrainingConfig.sample_reweighting[sample_name])
        sig_aux_data_test.append(cur_aux_data)

    # load all background processes
    bkg_data_test = [] # this holds all the branches used as inputs to the classifier
    bkg_weights_test = []
    bkg_aux_data_test = [] # this holds some other branches that may be important
    for sample, sample_name in zip(data_bkg, bkg_samples):
        cur_length = len(sample)
        sample = sample.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample
        cur_test = sample[int(data_slice[0] * cur_length) : int(data_slice[1] * cur_length)]
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights

        cur_aux_data = cur_test[TrainingConfig.auxiliary_branches].values
        bkg_data_test.append(cur_testdata)
        bkg_weights_test.append(cur_weights / slice_size * TrainingConfig.sample_reweighting[sample_name])
        bkg_aux_data_test.append(cur_aux_data)

    # also prepare the total, concatenated versions
    data_test = sig_data_test + bkg_data_test
    aux_test = sig_aux_data_test + bkg_aux_data_test
    weights_test = sig_weights_test + bkg_weights_test
    samples = sig_samples + bkg_samples

    # prepare the common mBB binning for all signal regions
    SR_low = 30
    SR_up = 210
    SR_binwidth = 10
    SR_binning = np.linspace(SR_low, SR_up, num = 1 + int((SR_up - SR_low) / SR_binwidth), endpoint = True)

    effdict = {}

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

    # compute the total number of available signal events
    sig_events_total_2j = inclusive_2J.get_number_events("Hbb")
    sig_events_total_3j = inclusive_3J.get_number_events("Hbb")

    print("total 2J signal events: {}".format(sig_events_total_2j))
    print("total 3J signal events: {}".format(sig_events_total_3j))

    # fill the cut-based categories
    for cur_nJ, sig_events_total in zip([2, 3], [sig_events_total_2j, sig_events_total_3j]):
        # first, export the categories of the cut-based analysis: high / low MET
        low_MET_cat = CutBasedCategoryFiller.create_low_MET_category(process_events = data_test,
                                                                     process_aux_events = aux_test,
                                                                     process_weights = weights_test,
                                                                     process_names = samples,
                                                                     nJ = cur_nJ
                                                                 )

        low_MET_cat.export_ROOT_histogram(binning = SR_binning, processes = sig_samples + bkg_samples, var_names = "mBB",
                                          outfile_path = os.path.join(outdir, "{}jet_low_MET.root".format(cur_nJ)), clipping = True, density = False)

        CategoryPlotter.plot_category_composition(low_MET_cat, binning = SR_binning, outpath = os.path.join(outdir, "{}jet_low_MET.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", r'150 GeV < MET < 200 GeV', "dRBB < 1.8", "nJ = {}".format(cur_nJ)], args = {})

        # get the signal efficiency for this category
        sigeff = low_MET_cat.get_number_events("Hbb") / sig_events_total
        effdict["low_MET_{}J".format(cur_nJ)] = sigeff

        high_MET_cat = CutBasedCategoryFiller.create_high_MET_category(process_events = data_test,
                                                                       process_aux_events = aux_test,
                                                                       process_weights = weights_test,
                                                                       process_names = samples,
                                                                       nJ = cur_nJ
                                                                   )

        high_MET_cat.export_ROOT_histogram(binning = SR_binning, processes = sig_samples + bkg_samples, var_names = "mBB",
                                          outfile_path = os.path.join(outdir, "{}jet_high_MET.root".format(cur_nJ)), clipping = True, density = False)

        CategoryPlotter.plot_category_composition(high_MET_cat, binning = SR_binning, outpath = os.path.join(outdir, "{}jet_high_MET.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MC16d", "MET > 200 GeV", "dRBB < 1.2", "nJ = {}".format(cur_nJ)], args = {})

        # get the signal efficiency for this category
        sigeff = high_MET_cat.get_number_events("Hbb") / sig_events_total
        effdict["high_MET_{}J".format(cur_nJ)] = sigeff

    return effdict

if __name__ == "__main__":
    parser = ArgumentParser(description = "optimize the cuts in the CBA for maximum binned significance")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    args = vars(parser.parse_args())

    effs = GetCBASignalEfficiencies(**args)
    for name, value in effs.items():
        print("{}: {}".format(name, value))
