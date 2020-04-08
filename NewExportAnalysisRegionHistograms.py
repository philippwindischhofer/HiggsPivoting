import pandas as pd
import numpy as np
import os, pickle
from argparse import ArgumentParser

from NewTrainAdversarialModel import extract_shuffled_slice
from base.Configs import TrainingConfig
from models.ModelCollection import ModelCollection
from analysis.NewCutBasedCategoryFiller import CutBasedCategoryFiller
from analysis.NewClassifierBasedCategoryFiller import ClassifierBasedCategoryFiller
from plotting.CategoryPlotter import CategoryPlotter
from plotting.ModelEvaluator import ModelEvaluator
from plotting.TrainingStatisticsPlotter import TrainingStatisticsPlotter

def renormalize_yields(sample, factor):
    sample["EventWeight"] *= factor
    return sample

def ExportAnalysisRegionHistograms(infile_path, model_dir, out_dir):
    
    # load the test dataset
    sig_sample_names = TrainingConfig.sig_samples
    bkg_sample_names = TrainingConfig.bkg_samples
    test_slice = TrainingConfig.test_slice
    test_slice_size = test_slice[1] - test_slice[0]

    print("loading data ...")
    sig_data = [pd.read_hdf(infile_path, key = cur_sample) for cur_sample in sig_sample_names]
    bkg_data = [pd.read_hdf(infile_path, key = cur_sample) for cur_sample in bkg_sample_names]
    print("done!")

    sig_data_test = [extract_shuffled_slice(cur_sample, slice_def = test_slice) for cur_sample in sig_data]
    bkg_data_test = [extract_shuffled_slice(cur_sample, slice_def = test_slice) for cur_sample in bkg_data]

    # take care to preserve the total event yield
    sig_data_test = [renormalize_yields(cur_sample, factor = 1.0 / test_slice_size) for cur_sample in sig_data_test]
    bkg_data_test = [renormalize_yields(cur_sample, factor = 1.0 / test_slice_size) for cur_sample in bkg_data_test]

    all_processes = sig_data_test + bkg_data_test
    all_process_names = sig_sample_names + bkg_sample_names

    # load the model
    mcoll = ModelCollection.from_config(model_dir)

    # some settings concerning cuts and binning
    SR_low = 30
    SR_up = 210
    SR_binwidth = 10
    SR_binning = np.linspace(SR_low, SR_up, num = 1 + int((SR_up - SR_low) / SR_binwidth), endpoint = True)

    cuts = {2: [0.0, 0.3936688696975736, 0.9162186612913272],
            3: [0.0, 0.35975037002858584, 0.861855992060236]}

    cut_labels = ["tight", "loose"]

    CBA_original = {"MET_cut": 200, "dRBB_highMET_cut": 1.2, "dRBB_lowMET_cut": 1.8}
    CBA_optimized = {"MET_cut": 191, "dRBB_highMET_cut": 1.2, "dRBB_lowMET_cut": 5.0}

    adversary_label = "temp"

    # initially, make some plots showing the evolution of some training metrics
    for model in mcoll.models:
        training_plotter = TrainingStatisticsPlotter(model.path)
        training_plotter.plot(model.path)

    # fill inclusive categories with 2j / 3j events
    inclusive_2J = CutBasedCategoryFiller.create_nJ_category(process_data = all_processes, process_names = all_process_names, nJ = 2)

    for cur_process in all_process_names:
        inclusive_2J.export_histogram(binning = SR_binning, processes = [cur_process], var_name = "mBB", outfile = os.path.join(out_dir, "dist_mBB_{}_2jet.pkl".format(cur_process)), density = True)

    inclusive_2J.export_histogram(binning = SR_binning, processes = bkg_sample_names, var_name = "mBB", outfile = os.path.join(out_dir, "dist_mBB_bkg_2jet.pkl"), density = True)

    inclusive_3J = CutBasedCategoryFiller.create_nJ_category(process_data = all_processes, process_names = all_process_names, nJ = 3)

    for cur_process in all_process_names:
        inclusive_3J.export_histogram(binning = SR_binning, processes = [cur_process], var_name = "mBB", outfile = os.path.join(out_dir, "dist_mBB_{}_3jet.pkl".format(cur_process)), density = True)

    inclusive_3J.export_histogram(binning = SR_binning, processes = bkg_sample_names, var_name = "mBB", outfile = os.path.join(out_dir, "dist_mBB_bkg_3jet.pkl"), density = True)

    anadict = {}

    # fill the signal regions of the classifier-based analysis
    #for nJ, cur_inclusive_cat in zip([2, 3], [inclusive_2J, inclusive_3J]):
    for nJ, cur_inclusive_cat in zip([3], [inclusive_3J]):
    #for nJ, cur_inclusive_cat in zip([2], [inclusive_2J]):
        for cut_end, cut_start, cut_label in zip(cuts[nJ][0:-1], cuts[nJ][1:], cut_labels):
            print("exporting {}J region with sigeff range {} - {}".format(nJ, cut_start, cut_end))

            cur_cat = ClassifierBasedCategoryFiller.create_classifier_category(mcoll, sig_process_data = sig_data_test, sig_process_names = sig_sample_names,
                                                                               bkg_process_data = bkg_data_test, bkg_process_names = bkg_sample_names,
                                                                               classifier_sigeff_range = (cut_start, cut_end), nJ = nJ)
            
            cur_cat.export_ROOT_histogram(binning = SR_binning, processes = all_process_names, var_names = "mBB",
                                          outfile_path = os.path.join(out_dir, "region_{}jet_{}_{}.root".format(nJ, cut_start, cut_end)), clipping = True, density = False)

            for cur_process in all_process_names:
                cur_cat.export_histogram(binning = SR_binning, processes = [cur_process], var_name = "mBB", outfile = os.path.join(out_dir, "dist_mBB_{}_{}jet_{}.pkl".format(cur_process, nJ, cut_label)), density = True)

            anadict["{}_{}jet_sig_eff".format(cut_label, nJ)] = ModelEvaluator.get_efficiency(cur_cat, cur_inclusive_cat, sig_sample_names)
            anadict["{}_{}jet_bkg_eff".format(cut_label, nJ)] = ModelEvaluator.get_efficiency(cur_cat, cur_inclusive_cat, bkg_sample_names)

            anadict["{}_{}jet_inv_JS_bkg".format(cut_label, nJ)] = 1.0 / ModelEvaluator.get_JS_categories(cur_cat, cur_inclusive_cat, binning = SR_binning, var = "mBB", processes = bkg_sample_names)
            anadict["{}_{}jet_binned_sig".format(cut_label, nJ)] = cur_cat.get_binned_significance(binning = SR_binning, signal_processes = sig_sample_names, background_processes = bkg_sample_names, var_name = "mBB")

            cur_cat.export_histogram(binning = SR_binning, processes = bkg_sample_names, var_name = "mBB", outfile = os.path.join(out_dir, "dist_mBB_bkg_{}jet_{}.pkl".format(nJ, cut_label)), density = True)

            CategoryPlotter.plot_category_composition(cur_cat, binning = SR_binning, outpath = os.path.join(out_dir, "dist_mBB_region_{}jet_{}_{}.pdf".format(nJ, cut_start, cut_end)), 
                                                      var = "mBB", xlabel = r'$m_{bb}$ [GeV]', plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', cut_label + r', {} jet'.format(nJ), adversary_label])

            CategoryPlotter.plot_category_composition(cur_cat, binning = SR_binning, outpath = os.path.join(out_dir, "dist_mBB_region_{}jet_{}_{}_nostack.pdf".format(nJ, cut_start, cut_end)), 
                                                      var = "mBB", xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.", plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', cut_label + r', {} jet'.format(nJ), adversary_label], stacked = False, histtype = 'step', density = True)


    # fill the signal regions of the cut-based analysis
    for nJ, cur_inclusive_cat in zip([2, 3], [inclusive_2J, inclusive_3J]):
        for cur_cuts, prefix in zip([CBA_original, CBA_optimized], ["original_", "optimized_"]):

            # low-MET regions
            print("filling {} jet low_MET category".format(nJ))
            low_MET_cat = CutBasedCategoryFiller.create_low_MET_category(process_data = all_processes, process_names = all_process_names, nJ = nJ, cuts = cur_cuts)

            low_MET_cat.export_ROOT_histogram(binning = SR_binning, processes = all_process_names, var_names = "mBB",
                                              outfile_path = os.path.join(out_dir, prefix + "{}jet_low_MET.root".format(nJ)), clipping = True, density = False)

            for cur_process in all_process_names:
                low_MET_cat.export_histogram(binning = SR_binning, processes = [cur_process], var_name = "mBB", outfile = os.path.join(out_dir, prefix + "dist_mBB_{}_{}jet_low_MET.pkl".format(cur_process, nJ)), density = True)

            low_MET_cat.export_histogram(binning = SR_binning, processes = bkg_sample_names, var_name = "mBB", outfile = os.path.join(out_dir, prefix + "dist_mBB_bkg_{}jet_low_MET.pkl".format(nJ)), density = True)

            anadict[prefix + "low_MET_{}jet_sig_eff".format(nJ)] = ModelEvaluator.get_efficiency(low_MET_cat, cur_inclusive_cat, sig_sample_names)
            anadict[prefix + "low_MET_{}jet_bkg_eff".format(nJ)] = ModelEvaluator.get_efficiency(low_MET_cat, cur_inclusive_cat, bkg_sample_names)
            
            anadict[prefix + "low_MET_{}jet_inv_JS_bkg".format(nJ)] = 1.0 / ModelEvaluator.get_JS_categories(low_MET_cat, cur_inclusive_cat, binning = SR_binning, var = "mBB", processes = bkg_sample_names)
            anadict[prefix + "low_MET_{}jet_binned_sig".format(nJ)] = low_MET_cat.get_binned_significance(binning = SR_binning, signal_processes = sig_sample_names, background_processes = bkg_sample_names, var_name = "mBB")
            
            CategoryPlotter.plot_category_composition(low_MET_cat, binning = SR_binning, outpath = os.path.join(out_dir, prefix + "{}jet_low_MET.pdf".format(nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                      plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', r'150 GeV < $E_{\mathrm{T}}^{\mathrm{miss}}$' + '< {MET_cut} GeV'.format(**cur_cuts), r'$\Delta R_{{bb}} < {dRBB_lowMET_cut}$'.format(**cur_cuts), r'{} jet'.format(nJ)], args = {})
            
            CategoryPlotter.plot_category_composition(low_MET_cat, binning = SR_binning, outpath = os.path.join(out_dir, prefix + "{}jet_low_MET_nostack.pdf".format(nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.",
                                                      plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', r'150 GeV < $E_{\mathrm{T}}^{\mathrm{miss}}$' +  '< {MET_cut} GeV'.format(**cur_cuts), r'$\Delta R_{{bb}} < {dRBB_lowMET_cut}$'.format(**cur_cuts), r'{} jet'.format(nJ)], args = {}, stacked = False, histtype = 'step', density = True)
            
            # high-MET regions
            print("filling {} jet high_MET category".format(nJ))
            high_MET_cat = CutBasedCategoryFiller.create_high_MET_category(process_data = all_processes, process_names = all_process_names, nJ = nJ, cuts = cur_cuts)

            high_MET_cat.export_ROOT_histogram(binning = SR_binning, processes = all_process_names, var_names = "mBB",
                                              outfile_path = os.path.join(out_dir, prefix + "{}jet_high_MET.root".format(nJ)), clipping = True, density = False)

            for cur_process in all_process_names:
                high_MET_cat.export_histogram(binning = SR_binning, processes = [cur_process], var_name = "mBB", outfile = os.path.join(out_dir, prefix + "dist_mBB_{}_{}jet_high_MET.pkl".format(cur_process, nJ)), density = True)

            high_MET_cat.export_histogram(binning = SR_binning, processes = bkg_sample_names, var_name = "mBB", outfile = os.path.join(out_dir, prefix + "dist_mBB_bkg_{}jet_high_MET.pkl".format(nJ)), density = True)

            anadict[prefix + "high_MET_{}jet_sig_eff".format(nJ)] = ModelEvaluator.get_efficiency(high_MET_cat, cur_inclusive_cat, sig_sample_names)
            anadict[prefix + "high_MET_{}jet_bkg_eff".format(nJ)] = ModelEvaluator.get_efficiency(high_MET_cat, cur_inclusive_cat, bkg_sample_names)
            
            anadict[prefix + "high_MET_{}jet_inv_JS_bkg".format(nJ)] = 1.0 / ModelEvaluator.get_JS_categories(high_MET_cat, cur_inclusive_cat, binning = SR_binning, var = "mBB", processes = bkg_sample_names)
            anadict[prefix + "high_MET_{}jet_binned_sig".format(nJ)] = high_MET_cat.get_binned_significance(binning = SR_binning, signal_processes = sig_sample_names, background_processes = bkg_sample_names, var_name = "mBB")
            
            CategoryPlotter.plot_category_composition(high_MET_cat, binning = SR_binning, outpath = os.path.join(out_dir, prefix + "{}jet_high_MET.pdf".format(nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                                      plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', r'$E_{\mathrm{T}}^{\mathrm{miss}}$ >' + ' {MET_cut} GeV'.format(**cur_cuts), r'$\Delta R_{{bb}} < {dRBB_highMET_cut}$'.format(**cur_cuts), r'{} jet'.format(nJ)], args = {})
            
            CategoryPlotter.plot_category_composition(high_MET_cat, binning = SR_binning, outpath = os.path.join(out_dir, prefix + "{}jet_high_MET_nostack.pdf".format(nJ)), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.",
                                                      plotlabel = ["MadGraph + Pythia8", r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$', r'$E_{\mathrm{T}}^{\mathrm{miss}}$ >' + ' {MET_cut} GeV'.format(**cur_cuts), r'$\Delta R_{{bb}} < {dRBB_highMET_cut}$'.format(**cur_cuts), r'{} jet'.format(nJ)], args = {}, stacked = False, histtype = 'step', density = True)            

    anadict.update(mcoll.models[0].create_paramdict())
    print("got the following anadict: {}".format(anadict))
    with open(os.path.join(out_dir, "anadict.pkl"), "wb") as outfile:
        pickle.dump(anadict, outfile)

if __name__ == "__main__":
    parser = ArgumentParser(description = "populate analysis signal regions and export them to be used with HistFitter")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--model_dir", action = "store", dest = "model_dir")
    parser.add_argument("--out_dir", action = "store", dest = "out_dir")
    args = vars(parser.parse_args())
    
    ExportAnalysisRegionHistograms(**args)
