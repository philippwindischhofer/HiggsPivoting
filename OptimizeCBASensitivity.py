import os, pickle
import numpy as np
import pandas as pd
import subprocess as sp
import itertools
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization
from argparse import ArgumentParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from base.Configs import TrainingConfig
from analysis.CutBasedCategoryFiller import CutBasedCategoryFiller
from DatasetExtractor import TrainNuisAuxSplit

evalcnt = 0

def EvaluateAsimovSignificance(process_events, process_aux_events, process_weights, process_names, signal_process_names, background_process_names, binning, cuts, fit_dir):
    global evalcnt
    evalcnt += 1

    if not os.path.exists(fit_dir):
        os.makedirs(fit_dir)

    sensdict = {}

    # first, fill the four event categories (high / low MET, each split into 2 jet and 3 jet) and export them
    low_MET_cat_2j = CutBasedCategoryFiller.create_low_MET_category(process_events = process_events,
                                                                    process_aux_events = process_aux_events,
                                                                    process_weights = process_weights,
                                                                    process_names = process_names,
                                                                    nJ = 2, cuts = cuts)
    low_MET_cat_2j.export_ROOT_histogram(binning = binning, processes = process_names, var_names = "mBB",
                                         outfile_path = os.path.join(fit_dir, "2jet_low_MET.root"), clipping = True, density = False)
    
    high_MET_cat_2j = CutBasedCategoryFiller.create_high_MET_category(process_events = process_events,
                                                                   process_aux_events = process_aux_events,
                                                                   process_weights = process_weights,
                                                                   process_names = process_names,
                                                                   nJ = 2, cuts = cuts)
    high_MET_cat_2j.export_ROOT_histogram(binning = binning, processes = process_names, var_names = "mBB",
                                         outfile_path = os.path.join(fit_dir, "2jet_high_MET.root"), clipping = True, density = False)

    low_MET_cat_3j = CutBasedCategoryFiller.create_low_MET_category(process_events = process_events,
                                                                    process_aux_events = process_aux_events,
                                                                    process_weights = process_weights,
                                                                    process_names = process_names,
                                                                    nJ = 3, cuts = cuts)
    low_MET_cat_3j.export_ROOT_histogram(binning = binning, processes = process_names, var_names = "mBB",
                                         outfile_path = os.path.join(fit_dir, "3jet_low_MET.root"), clipping = True, density = False)

    high_MET_cat_3j = CutBasedCategoryFiller.create_high_MET_category(process_events = process_events,
                                                                      process_aux_events = process_aux_events,
                                                                      process_weights = process_weights,
                                                                      process_names = process_names,
                                                                      nJ = 3, cuts = cuts)
    high_MET_cat_3j.export_ROOT_histogram(binning = binning, processes = process_names, var_names = "mBB",
                                          outfile_path = os.path.join(fit_dir, "3jet_high_MET.root"), clipping = True, density = False)

    # prepare the script to run HistFitter and extract the result
    hist_fitter_sceleton = """#!/bin/bash
    export PYTHONPATH=/home/windischhofer/HiggsPivoting/fitting:$PYTHONPATH
    export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
    export ALRB_rootVersion=6.14.04-x86_64-slc6-gcc62-opt
    source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
    lsetup root
    source /home/windischhofer/HistFitter/v0.61.0/setup.sh
    cd {fit_dir}
    /home/windischhofer/HistFitter/v0.61.0/scripts/HistFitter.py -w -f -F disc -m ALL -a -z --userArg {fit_dir} /home/windischhofer/HiggsPivoting/fitting/ShapeFitHighLowMETBackgroundFloating.py &> {fit_log}
    python /home/windischhofer/HiggsPivoting/fitting/ConvertHistFitterResult.py --mode hypotest --infile {fit_outfile} --outkey asimov_sig_high_low_MET_background_floating --outfile {sig_outfile}

    rm -r {fit_dir}/config
    rm -r {fit_dir}/data
    rm -r {fit_dir}/results
    """

    # run the fit
    pars = {"fit_dir": fit_dir, "fit_log": os.path.join(fit_dir, "fit.log"), 
            "fit_outfile": os.path.join(fit_dir, "results", "ShapeFitHighLowMETBackgroundFloating_hypotest.root"),
            "sig_outfile": os.path.join(fit_dir, "fit_results.pkl")}

    jobfile_path = os.path.join(fit_dir, "do_fit.sh")
    with open(jobfile_path, "w") as jobfile:
        jobfile.write(hist_fitter_sceleton.format(**pars))    

    sp.check_output(["sh", jobfile_path])

    # read the result
    with open(os.path.join(fit_dir, "fit_results.pkl"), "rb") as infile:
        resdict = pickle.load(infile)
        sensdict["combined"] = resdict["asimov_sig_high_low_MET_background_floating"]

    print("testing: MET_cut = {}, dRBB_highMET_cut = {}, dRBB_lowMET_cut = {} --> {} sigma".format(
        cuts["MET_cut"], cuts["dRBB_highMET_cut"], cuts["dRBB_lowMET_cut"], sensdict["combined"])
    )

    # update the log
    fit_evolution_file = os.path.join(fit_dir, "fit_evolution.pkl")
    try:
        with open(fit_evolution_file, "rb") as evol_logfile:
            fit_evolution_dict = pickle.load(evol_logfile)
    except:
        fit_evolution_dict = {}

    print("current evolution log: {}".format(fit_evolution_dict))
    fit_evolution_dict[evalcnt] = sensdict

    with open(fit_evolution_file, "wb") as evol_logfile:
        pickle.dump(fit_evolution_dict, evol_logfile)

    return sensdict

def OptimizeCBASensitivity(infile_path, outdir, do_plots = True):
    data_slice = TrainingConfig.training_slice
    slice_size = data_slice[1] - data_slice[0]

    # read the test dataset, which will be used to get the expected sensitivity of the analysis
    sig_samples = TrainingConfig.sig_samples
    bkg_samples = TrainingConfig.bkg_samples

    print("loading data ...")
    sig_data = [pd.read_hdf(infile_path, key = sig_sample) for sig_sample in sig_samples]
    bkg_data = [pd.read_hdf(infile_path, key = bkg_sample) for bkg_sample in bkg_samples]

    sig_data_train = []
    sig_mBB_train = []
    sig_weights_train = []
    sig_aux_data_train = []
    for sample in sig_data:
        cur_length = len(sample)
        sample = sample.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample
        cur_train = sample[int(data_slice[0] * cur_length) : int(data_slice[1] * cur_length)]
        cur_traindata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_train) # load the standard classifier input, nuisances and weights
        cur_aux_data = cur_train[TrainingConfig.other_branches].values
        sig_data_train.append(cur_traindata)
        sig_mBB_train.append(cur_nuisdata)
        sig_weights_train.append(cur_weights / slice_size)
        sig_aux_data_train.append(cur_aux_data)

    bkg_data_train = []
    bkg_mBB_train = []
    bkg_weights_train = []
    bkg_aux_data_train = []
    for sample in bkg_data:
        cur_length = len(sample)
        sample = sample.sample(frac = 1, random_state = 12345).reset_index(drop = True) # shuffle the sample
        cur_train = sample[int(data_slice[0] * cur_length) : int(data_slice[1] * cur_length)]
        cur_traindata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_train) # load the standard classifier input, nuisances and weights
        cur_aux_data = cur_train[TrainingConfig.other_branches].values
        bkg_data_train.append(cur_traindata)
        bkg_mBB_train.append(cur_nuisdata)
        bkg_weights_train.append(cur_weights / slice_size)
        bkg_aux_data_train.append(cur_aux_data)

    # also prepare the total, concatenated versions
    data_train = sig_data_train + bkg_data_train
    aux_train = sig_aux_data_train + bkg_aux_data_train
    weights_train = sig_weights_train + bkg_weights_train
    samples = sig_samples + bkg_samples

    # define the SR binning for mBB
    SR_low = 30
    SR_up = 210
    SR_binwidth = 10
    SR_mBB_binning = np.linspace(SR_low, SR_up, num = 1 + int((SR_up - SR_low) / SR_binwidth), endpoint = True)

    print("mBB binning: {}".format(SR_mBB_binning))

    original_cuts = {"MET_cut": 200, "dRBB_highMET_cut": 1.2, "dRBB_lowMET_cut": 1.8}

    # the objective function that needs to be minimized
    costfunc = lambda cuts: -EvaluateAsimovSignificance(process_events = data_train, process_aux_events = aux_train, 
                                                        process_weights = weights_train, process_names = samples, 
                                                        signal_process_names = sig_samples, background_process_names = bkg_samples, 
                                                        binning = SR_mBB_binning, cuts = cuts, fit_dir = outdir)["combined"]
    
    costfunc_bayes = lambda MET_cut, dRBB_highMET_cut, dRBB_lowMET_cut: -costfunc({"MET_cut": MET_cut, "dRBB_highMET_cut": dRBB_highMET_cut, "dRBB_lowMET_cut": dRBB_lowMET_cut})
    
    # then, try a global search strategy
    ranges_bayes = {"MET_cut": (150, 250), "dRBB_highMET_cut": (0.5, 5.0), "dRBB_lowMET_cut": (0.5, 5.0)}
    gp_params = {'kernel': 1.0 * Matern(length_scale = 0.05, length_scale_bounds = (1e-1, 1e2), nu = 1.5)}
    optimizer = BayesianOptimization(
        f = costfunc_bayes,
        pbounds = ranges_bayes,
        random_state = None
    )
    optimizer.maximize(init_points = 20, n_iter = 1, acq = 'poi', kappa = 3, **gp_params)

    xi_scheduler = lambda iteration: 0.01 + 0.19 * np.exp(-0.004 * iteration)
    for it in range(400):
        cur_xi = xi_scheduler(it)
        print("using xi = {}".format(cur_xi))
        optimizer.maximize(init_points = 0, n_iter = 1, acq = 'poi', kappa = 3, xi = cur_xi, **gp_params)
    
    # print the results
    print("==============================================")
    print("initial cuts:")
    print("==============================================")
    print("MET_cut = {}".format(original_cuts["MET_cut"]))
    print("dRBB_highMET_cut = {}".format(original_cuts["dRBB_highMET_cut"]))
    print("dRBB_lowMET_cut = {}".format(original_cuts["dRBB_lowMET_cut"]))
    print("significance = {} sigma".format(costfunc_bayes(**original_cuts)))
    print("==============================================")

    print("==============================================")
    print("optimized cuts (global optimization):")
    print("==============================================")
    print("MET_cut = {}".format(optimizer.max["params"]["MET_cut"]))
    print("dRBB_highMET_cut = {}".format(optimizer.max["params"]["dRBB_highMET_cut"]))
    print("dRBB_lowMET_cut = {}".format(optimizer.max["params"]["dRBB_lowMET_cut"]))
    print("significance = {} sigma".format(optimizer.max["target"]))
    print("==============================================")

    # save the results:
    with open(os.path.join(outdir, "opt_results.pkl"), "wb") as opt_outfile:
        pickle.dump(optimizer.max, opt_outfile)
        
if __name__ == "__main__":
    parser = ArgumentParser(description = "optimize the cuts in the CBA for maximum binned significance")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    args = vars(parser.parse_args())

    outdir = args["outdir"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    OptimizeCBASensitivity(**args)
