import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from plotting.CategoryPlotter import CategoryPlotter
from analysis.Category import Category
from analysis.CutBasedCategoryFiller import CutBasedCategoryFiller
from base.Configs import TrainingConfig
from DatasetExtractor import TrainNuisAuxSplit

def get_binning(low, high, binwidth):
    return np.linspace(low, high, num = int((high - low) / binwidth), endpoint = True)

def MakeDistributionControlPlots(infile, outdir, test_size = 0.999):
    # sig_samples = ["Hbb"]
    # bkg_samples = ["ttbar", "Zjets", "Wjets", "diboson", "singletop"]

    # for MadGraph
    sig_samples = ["Hbb"]
    bkg_samples = ["ttbar", "Zjets", "Wjets"]

    samples = sig_samples + bkg_samples

    # set up proper binnings for different variables
    binnings = {}
    binnings["mBB"] = get_binning(30, 600, 10)
    binnings["dRBB"] = get_binning(0.0, 3.0, 0.1)
    binnings["pTB1"] = get_binning(0, 300, 10)
    binnings["pTB2"] = get_binning(0, 300, 10)
    binnings["MET"] = get_binning(0, 300, 10)
    binnings["dEtaBB"] = get_binning(0, 5, 0.1)
    binnings["dPhiMETdijet"] = get_binning(0, np.pi, 0.1)
    binnings["SumPtJet"] = get_binning(0, 500, 10)

    print("loading data ...")
    data = [pd.read_hdf(infile_path, key = sample) for sample in samples]

    for cur_df, sample in zip(data, samples):
        print("have {} events available for '{}'".format(len(cur_df), sample))

    data_test = []
    mBB_test = []
    weights_test = []
    aux_data_test = []
    for sample in data:
        _, cur_test = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights
        cur_aux_data = cur_test[TrainingConfig.other_branches].values
        data_test.append(cur_testdata)
        mBB_test.append(cur_nuisdata)
        weights_test.append(cur_weights / test_size)
        aux_data_test.append(cur_aux_data)

    # first, plot the total event content (i.e. corresponding to an "inclusive" event category)
    inclusive = Category("inclusive")
    for events, weights, process in zip(data_test, weights_test, samples):
        inclusive.add_events(events = events, weights = weights, process = process, event_variables = TrainingConfig.training_branches, aux_content = cur_aux_data, aux_variables = TrainingConfig.other_branches)

    # print total event numbers for all processes
    print("============================")
    print(" inclusive expected event yield ")
    print("============================")
    for process in samples:
        print("{}: {} events".format(process, inclusive.get_number_events(process)))
    print("============================")
        

    # also fill inclusive 2- and 3-jet categories to get a baseline for the shapes
    inclusive_2J = CutBasedCategoryFiller.create_nJ_category(process_events = data_test,
                                                             process_aux_events = aux_data_test,
                                                             process_weights = weights_test,
                                                             process_names = samples,
                                                             nJ = 2)

    print("============================")
    print(" inclusive 2j expected event yield ")
    print("============================")
    for process in samples:
        print("{}: {} events".format(process, inclusive_2J.get_number_events(process)))
    print("============================")

    inclusive_3J = CutBasedCategoryFiller.create_nJ_category(process_events = data_test,
                                                             process_aux_events = aux_data_test,
                                                             process_weights = weights_test,
                                                             process_names = samples,
                                                             nJ = 3)

    print("============================")
    print(" inclusive 3j expected event yield ")
    print("============================")
    for process in samples:
        print("{}: {} events".format(process, inclusive_3J.get_number_events(process)))
    print("============================")
    
    # now, create separate histograms for each process and each event variable
    for cur_var in TrainingConfig.training_branches:
        for cur_process in samples:
            CategoryPlotter.plot_category_composition(inclusive, binning = binnings[cur_var], outpath = os.path.join(outdir, "dist_{}_{}_inclusive.pdf".format(cur_var, cur_process)), var = cur_var, 
                                                      process_order = [cur_process], xlabel = cur_var, plotlabel = ["inclusive"], args = {})
            inclusive.export_histogram(binning = binnings[cur_var], processes = [cur_process], var_name = cur_var, 
                                       outfile = os.path.join(outdir, "dist_{}_{}_inclusive.pkl".format(cur_var, cur_process)), clipping = True, density = True)

            CategoryPlotter.plot_category_composition(inclusive_2J, binning = binnings[cur_var], outpath = os.path.join(outdir, "dist_{}_{}_inclusive_2J.pdf".format(cur_var, cur_process)), var = cur_var, 
                                                      process_order = [cur_process], xlabel = cur_var, plotlabel = ["inclusive, nJ = 2"], args = {})
            inclusive_2J.export_histogram(binning = binnings[cur_var], processes = [cur_process], var_name = cur_var, 
                                       outfile = os.path.join(outdir, "dist_{}_{}_inclusive_2J.pkl".format(cur_var, cur_process)), clipping = True, density = True)

            CategoryPlotter.plot_category_composition(inclusive_3J, binning = binnings[cur_var], outpath = os.path.join(outdir, "dist_{}_{}_inclusive_3J.pdf".format(cur_var, cur_process)), var = cur_var, 
                                                      process_order = [cur_process], xlabel = cur_var, plotlabel = ["inclusive, nJ = 3"], args = {})
            inclusive_3J.export_histogram(binning = binnings[cur_var], processes = [cur_process], var_name = cur_var, 
                                          outfile = os.path.join(outdir, "dist_{}_{}_inclusive_3J.pkl".format(cur_var, cur_process)), clipping = True, density = True)

if __name__ == "__main__":
    parser = ArgumentParser(description = "generate distributions of event variables to check their integrity")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    args = vars(parser.parse_args())

    infile_path = args["infile_path"]
    outdir = args["outdir"]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    MakeDistributionControlPlots(infile_path, outdir)
