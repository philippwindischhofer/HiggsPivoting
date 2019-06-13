import os, pickle
from argparse import ArgumentParser
import pandas as pd
import numpy as np

from models.AdversarialEnvironment import AdversarialEnvironment
from analysis.Category import Category
from analysis.ClassifierBasedCategoryFiller import ClassifierBasedCategoryFiller
from analysis.CutBasedCategoryFiller import CutBasedCategoryFiller
from base.Configs import TrainingConfig
from plotting.CategoryPlotter import CategoryPlotter
from plotting.ModelEvaluator import ModelEvaluator
from DatasetExtractor import TrainNuisAuxSplit

def main():
    parser = ArgumentParser(description = "populate analysis signal regions and export them to be used with HistFitter")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--model_dir", action = "store", dest = "model_dir")
    parser.add_argument("--out_dir", action = "store", dest = "out_dir")
    parser.add_argument("--use_test", action = "store_const", const = True, default = False)
    args = vars(parser.parse_args())

    # extract the validation or test dataset
    if args["use_test"]:
        print("using test dataset")
        data_slice = TrainingConfig.test_slice
    else:
        print("using validation dataset")
        data_slice = TrainingConfig.validation_slice

    slice_size = data_slice[1] - data_slice[0]

    infile_path = args["infile_path"]
    model_dir = args["model_dir"]
    outdir = args["out_dir"]

    sig_samples = TrainingConfig.sig_samples
    bkg_samples = TrainingConfig.bkg_samples

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
        sig_weights_test.append(cur_weights / slice_size)
        sig_aux_data_test.append(cur_aux_data)

    # also need to keep separate all signal events with 2 jets / 3 jets
    sig_data_test_2j = []
    sig_weights_test_2j = []
    sig_aux_data_test_2j = []

    sig_data_test_3j = []
    sig_weights_test_3j = []
    sig_aux_data_test_3j = []

    for sample, sample_name in zip(data_sig, sig_samples):
        cur_length = len(sample)
        sample = sample.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample
        cur_test = sample[int(data_slice[0] * cur_length) : int(data_slice[1] * cur_length)]
        cur_test = cur_test[cur_test["nJ"] == 2]
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights

        cur_aux_data = cur_test[TrainingConfig.auxiliary_branches].values
        sig_data_test_2j.append(cur_testdata)
        sig_weights_test_2j.append(cur_weights / slice_size)
        sig_aux_data_test_2j.append(cur_aux_data)

    for sample, sample_name in zip(data_sig, sig_samples):
        cur_length = len(sample)
        sample = sample.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample
        cur_test = sample[int(data_slice[0] * cur_length) : int(data_slice[1] * cur_length)]
        cur_test = cur_test[cur_test["nJ"] == 3]
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights

        cur_aux_data = cur_test[TrainingConfig.auxiliary_branches].values
        sig_data_test_3j.append(cur_testdata)
        sig_weights_test_3j.append(cur_weights / slice_size)
        sig_aux_data_test_3j.append(cur_aux_data)

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
        bkg_weights_test.append(cur_weights / slice_size)
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
    #SR_up = 600
    SR_binwidth = 10
    SR_binning = np.linspace(SR_low, SR_up, num = 1 + int((SR_up - SR_low) / SR_binwidth), endpoint = True)

    # also prepare the binning along the MVA dimension
    sigeff_binning = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0]

    # now convert the binning in terms of signal efficiency into the actual binning
    # in terms of the classifier output value
    #MVA_binning = [ClassifierBasedCategoryFiller._sigeff_to_score(env = env, signal_events = sig_data_test, signal_weights = sig_weights_test, sigeff = sigeff) for sigeff in sigeff_binning[::-1]]

    print("signal efficiency binning: {}".format(sigeff_binning))
    #print("classifier output binning: {}".format(MVA_binning))
    print("mBB binning: {}".format(SR_binning))

    # the cuts on the classifier (in terms of signal efficiency, separately for 2- and 3-jet events)
    # cuts = {2: [0.0, 0.3609901582980384, 0.8797635831120305], 
    #         3: [0.0, 0.32613469848310933, 0.7895892560956852]}

    # for CMS MC
    # cuts = {2: [0.0, 0.36099122139989426, 0.8802990977284879],
    #         3: [0.0, 0.326514026005491, 0.7917872994832997]}

    # for ATLAS MC
    # cuts = {2: [0.0, 0.34669408038894933, 0.9073610781343515],
    #         3: [0.0, 0.32230881360286623, 0.8215582540701756]}

    # for ATLAS MC (with optimized CBA)
    cuts = {2: [0.0, 0.40980884408751117, 0.9060291803032521],
            3: [0.0, 0.3726996799444123, 0.8493024023704923]}

    cut_labels = ["tight", "loose"]
    
    print("using the following cuts:")
    print(cuts)

    # fill the inclusive categories with 2j / 3j events
    inclusive_2J = CutBasedCategoryFiller.create_nJ_category(process_events = data_test,
                                                             process_aux_events = aux_test,
                                                             process_weights = weights_test,
                                                             process_names = samples,
                                                             nJ = 2)
    for cur_process in samples:
        inclusive_2J.export_histogram(binning = SR_binning, processes = [cur_process], var_name = "mBB", outfile = os.path.join(outdir, "dist_mBB_{}_2jet.pkl".format(cur_process)), density = True)

    inclusive_3J = CutBasedCategoryFiller.create_nJ_category(process_events = data_test,
                                                             process_aux_events = aux_test,
                                                             process_weights = weights_test,
                                                             process_names = samples,
                                                             nJ = 3)
    for cur_process in samples:
        inclusive_2J.export_histogram(binning = SR_binning, processes = [cur_process], var_name = "mBB", outfile = os.path.join(outdir, "dist_mBB_{}_3jet.pkl".format(cur_process)), density = True)

    total_events = inclusive_2J.get_total_events() + inclusive_3J.get_total_events()
    CBA_used_events = 0
    PCA_used_events = 0

    anadict = {}
    
    for cur_nJ, cur_inclusive_cat, cur_signal_events, cur_signal_weights, cur_signal_aux_events in zip([2, 3], [inclusive_2J, inclusive_3J], [sig_data_test_2j, sig_data_test_3j], [sig_weights_test_2j, sig_weights_test_3j], [sig_aux_data_test_2j, sig_aux_data_test_3j]):
        # first, export the categories of the cut-based analysis: high / low MET
        print("filling {} jet low_MET category".format(cur_nJ))
        low_MET_cat = CutBasedCategoryFiller.create_low_MET_category(process_events = data_test,
                                                                     process_aux_events = aux_test,
                                                                     process_weights = weights_test,
                                                                     process_names = samples,
                                                                     nJ = cur_nJ)
        print("filled {} signal events".format(low_MET_cat.get_number_events("Hbb")))

        low_MET_cat.export_ROOT_histogram(binning = SR_binning, processes = sig_samples + bkg_samples, var_names = "mBB",
                                          outfile_path = os.path.join(outdir, "{}jet_low_MET.root".format(cur_nJ)), clipping = True, density = False)

        anadict["low_MET_{}jet_sig_eff".format(cur_nJ)] = ModelEvaluator.get_efficiency(low_MET_cat, cur_inclusive_cat, sig_samples)
        anadict["low_MET_{}jet_bkg_eff".format(cur_nJ)] = ModelEvaluator.get_efficiency(low_MET_cat, cur_inclusive_cat, bkg_samples)

        anadict["low_MET_{}jet_inv_JS_bkg".format(cur_nJ)] = 1.0 / ModelEvaluator.get_JS_categories(low_MET_cat, cur_inclusive_cat, binning = SR_binning, var = "mBB", processes = bkg_samples)
        anadict["low_MET_{}jet_binned_sig".format(cur_nJ)] = low_MET_cat.get_binned_significance(binning = SR_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")

        CBA_used_events += low_MET_cat.get_total_events()

        for cur_process in samples:
            low_MET_cat.export_histogram(binning = SR_binning, processes = [cur_process], var_name = "mBB", outfile = os.path.join(outdir, "dist_mBB_{}_{}jet_low_MET.pkl".format(cur_process, cur_nJ)), density = True)

        CategoryPlotter.plot_category_composition(low_MET_cat, binning = SR_binning, outpath = os.path.join(outdir, "{}jet_low_MET.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', r'150 GeV < MET < 200 GeV', r'$\Delta R_{bb} < 1.8$', r'$n_J$ = {}'.format(cur_nJ)], args = {})

        CategoryPlotter.plot_category_composition(low_MET_cat, binning = SR_binning, outpath = os.path.join(outdir, "{}jet_low_MET_nostack.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.",
                                                  plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', r'150 GeV < MET < 200 GeV', r'$\Delta R_{bb} < 1.8$', r'$n_J$ = {}'.format(cur_nJ)], args = {}, stacked = False, histtype = 'step', density = True)

        print("filling {} jet high_MET category".format(cur_nJ))
        high_MET_cat = CutBasedCategoryFiller.create_high_MET_category(process_events = data_test,
                                                                       process_aux_events = aux_test,
                                                                       process_weights = weights_test,
                                                                       process_names = samples,
                                                                       nJ = cur_nJ)
        print("filled {} signal events".format(high_MET_cat.get_number_events("Hbb")))

        high_MET_cat.export_ROOT_histogram(binning = SR_binning, processes = sig_samples + bkg_samples, var_names = "mBB",
                                          outfile_path = os.path.join(outdir, "{}jet_high_MET.root".format(cur_nJ)), clipping = True, density = False)

        anadict["high_MET_{}jet_sig_eff".format(cur_nJ)] = ModelEvaluator.get_efficiency(high_MET_cat, cur_inclusive_cat, sig_samples)
        anadict["high_MET_{}jet_bkg_eff".format(cur_nJ)] = ModelEvaluator.get_efficiency(high_MET_cat, cur_inclusive_cat, bkg_samples)

        anadict["high_MET_{}jet_inv_JS_bkg".format(cur_nJ)] = 1.0 / ModelEvaluator.get_JS_categories(high_MET_cat, cur_inclusive_cat, binning = SR_binning, var = "mBB", processes = bkg_samples)
        anadict["high_MET_{}jet_binned_sig".format(cur_nJ)] = high_MET_cat.get_binned_significance(binning = SR_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")

        # compute JSD between the high-MET and low-MET categories
        anadict["{}jet_high_low_MET_inv_JS_bkg".format(cur_nJ)] = 1.0 / ModelEvaluator.get_JS_categories(high_MET_cat, low_MET_cat, binning = SR_binning, var = "mBB", processes = bkg_samples)
        anadict["{}jet_binned_sig_CBA".format(cur_nJ)] = (anadict["low_MET_{}jet_binned_sig".format(cur_nJ)]**2 + anadict["high_MET_{}jet_binned_sig".format(cur_nJ)]**2)**0.5

        CBA_used_events += high_MET_cat.get_total_events()

        for cur_process in samples:
            high_MET_cat.export_histogram(binning = SR_binning, processes = [cur_process], var_name = "mBB", outfile = os.path.join(outdir, "dist_mBB_{}_{}jet_high_MET.pkl".format(cur_process, cur_nJ)), density = True)

        CategoryPlotter.plot_category_composition(high_MET_cat, binning = SR_binning, outpath = os.path.join(outdir, "{}jet_high_MET.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                  plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', "MET > 200 GeV", r'$\Delta R_{bb} < 1.2$', r'$n_J$ = {}'.format(cur_nJ)], args = {})

        CategoryPlotter.plot_category_composition(high_MET_cat, binning = SR_binning, outpath = os.path.join(outdir, "{}jet_high_MET_nostack.pdf".format(cur_nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.",
                                                  plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', "MET > 200 GeV", r'$\Delta R_{bb} < 1.2$', r'$n_J$ = {}'.format(cur_nJ)], args = {}, stacked = False, histtype = 'step', density = True)

        # keep track of the tight and loose categories for later
        classifier_categories = {}

        # prepare N categories along the classifier output dimension
        for cut_end, cut_start, cut_label in zip(cuts[cur_nJ][0:-1], cuts[cur_nJ][1:], cut_labels):
            print("exporting {}J region with sigeff range {} - {}".format(cur_nJ, cut_start, cut_end))

            cur_cat = ClassifierBasedCategoryFiller.create_classifier_category(env,
                                                                               process_events = data_test,
                                                                               process_aux_events = aux_test,
                                                                               process_weights = weights_test,
                                                                               process_names = samples,
                                                                               signal_events = cur_signal_events,
                                                                               signal_weights = cur_signal_weights,
                                                                               signal_aux_events = cur_signal_aux_events,
                                                                               classifier_sigeff_range = (cut_start, cut_end),
                                                                               nJ = cur_nJ)
            cur_cat.export_ROOT_histogram(binning = SR_binning, processes = sig_samples + bkg_samples, var_names = "mBB",
                                           outfile_path = os.path.join(outdir, "region_{}jet_{}_{}.root".format(cur_nJ, cut_start, cut_end)), clipping = True, density = False)

            PCA_used_events += cur_cat.get_total_events()

            anadict["{}_{}jet_sig_eff".format(cut_label, cur_nJ)] = ModelEvaluator.get_efficiency(cur_cat, cur_inclusive_cat, sig_samples)
            anadict["{}_{}jet_bkg_eff".format(cut_label, cur_nJ)] = ModelEvaluator.get_efficiency(cur_cat, cur_inclusive_cat, bkg_samples)

            anadict["{}_{}jet_inv_JS_bkg".format(cut_label, cur_nJ)] = 1.0 / ModelEvaluator.get_JS_categories(cur_cat, cur_inclusive_cat, binning = SR_binning, var = "mBB", processes = bkg_samples)
            anadict["{}_{}jet_binned_sig".format(cut_label, cur_nJ)] = cur_cat.get_binned_significance(binning = SR_binning, signal_processes = sig_samples, background_processes = bkg_samples, var_name = "mBB")

            classifier_categories[cut_label] = cur_cat

            for cur_process in samples:
                cur_cat.export_histogram(binning = SR_binning, processes = [cur_process], var_name = "mBB", outfile = os.path.join(outdir, "dist_mBB_{}_{}jet_{}.pkl".format(cur_process, cur_nJ, cut_label)), density = True)

            CategoryPlotter.plot_category_composition(cur_cat, binning = SR_binning, outpath = os.path.join(outdir, "dist_mBB_region_{}jet_{}_{}.pdf".format(cur_nJ, cut_start, cut_end)), 
                                                      var = "mBB", xlabel = r'$m_{bb}$ [GeV]', plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', cut_label, r'$n_J$ = {}'.format(cur_nJ)])

            CategoryPlotter.plot_category_composition(cur_cat, binning = SR_binning, outpath = os.path.join(outdir, "dist_mBB_region_{}jet_{}_{}_nostack.pdf".format(cur_nJ, cut_start, cut_end)), 
                                                      var = "mBB", xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.", plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', cut_label, r'$n_J$ = {}'.format(cur_nJ)], stacked = False, histtype = 'step', density = True)

            print("filled {} signal events".format(cur_cat.get_number_events("Hbb")))

        # compute JSD between the tight and loose categories
        anadict["{}jet_tight_loose_inv_JS_bkg".format(cur_nJ)] = 1.0 / ModelEvaluator.get_JS_categories(classifier_categories["tight"], classifier_categories["loose"], binning = SR_binning, var = "mBB", processes = bkg_samples)
        anadict["{}jet_binned_sig_PCA".format(cur_nJ)] = (anadict["tight_{}jet_binned_sig".format(cur_nJ)]**2 + anadict["loose_{}jet_binned_sig".format(cur_nJ)]**2)**0.5

        # # now, also export the classifier categories for each jet split
        # class_cat_inclusive = ClassifierBasedCategoryFiller.create_classifier_category(env, 
        #                                                                                process_events = data_test,
        #                                                                                process_aux_events = aux_test,
        #                                                                                process_weights = weights_test,
        #                                                                                process_names = samples,
        #                                                                                signal_events = sig_data_test,
        #                                                                                signal_weights = sig_weights_test,
        #                                                                                signal_aux_events = sig_aux_data_test,
        #                                                                                classifier_sigeff_range = (1.0, 0.0),
        #                                                                                nJ = cur_nJ)

        # class_cat_inclusive.export_ROOT_histogram(binning = MVA_binning, processes = sig_samples + bkg_samples, var_names = "clf", 
        #                                           outfile_path = os.path.join(outdir, "{}jet_MVA.root".format(cur_nJ)), clipping = True, density = False, ignore_binning = True)

        # CategoryPlotter.plot_category_composition(class_cat_inclusive, binning = MVA_binning, outpath = os.path.join(outdir, "dist_MVA_{}J.pdf".format(cur_nJ)), var = "clf", xlabel = r'MVA', 
        #                                           plotlabel = ["MC16d", "MVA", "nJ = {}".format(cur_nJ)], logscale = True, ignore_binning = True)

        # CategoryPlotter.plot_category_composition(class_cat_inclusive, binning = MVA_binning, outpath = os.path.join(outdir, "dist_MVA_{}J_nostack.pdf".format(cur_nJ)), var = "clf", xlabel = r'MVA', ylabel = "a.u.",
        #                                           plotlabel = ["MC16d", "MVA", "nJ = {}".format(cur_nJ)], logscale = True, ignore_binning = True, stacked = False, histtype = 'step', density = True)

    print("event statistics:")
    print("have a total of {} events, CBA used {} events, ({}%)".format(total_events, CBA_used_events, CBA_used_events / total_events))
    print("have a total of {} events, PCA used {} events, ({}%)".format(total_events, PCA_used_events, PCA_used_events / total_events))

    anadict.update(env.create_paramdict())
    print("got the following anadict: {}".format(anadict))
    with open(os.path.join(outdir, "anadict.pkl"), "wb") as outfile:
        pickle.dump(anadict, outfile)

        
if __name__ == "__main__":
    main()
