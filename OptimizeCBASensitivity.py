import os
import numpy as np
import pandas as pd
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

def GenerateHeatMapSensitivityPlot(res, outfile, x_ticks, y_ticks, title = "", xlabel = "", ylabel = ""):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # convert the coordinates (essentially the 'bin centers') into proper tickmarks
    binwidth_x = x_ticks[1] - x_ticks[0]
    binwidth_y = y_ticks[1] - y_ticks[0]

    l_edges_x = list(x_ticks - binwidth_x / 2)
    u_edges_x = list(x_ticks + binwidth_x / 2)
    l_edges_y = list(y_ticks - binwidth_y / 2)
    u_edges_y = list(y_ticks + binwidth_y / 2)

    x_edges = sorted(l_edges_x + [u_edges_x[-1]])
    y_edges = sorted(l_edges_y + [u_edges_y[-1]])

    # prepare the colormap for the sensitivity
    cmap = plt.cm.coolwarm
    norm = mpl.colors.Normalize(vmin = 2.4, vmax = 2.8)

    im = ax.pcolor(x_edges, y_edges, res.T, cmap = cmap, norm = norm)
    cb = plt.colorbar(im)
    cb.set_label(r'binned significance [$\sigma$]')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)

def EvaluateBinnedSignificance(process_events, process_aux_events, process_weights, process_names, signal_process_names, background_process_names, binning, cuts):

    sensdict = {}

    # first, fill the four event categories (high / low MET, each split into 2 jet and 3 jet)
    low_MET_cat_2j = CutBasedCategoryFiller.create_low_MET_category(process_events = process_events,
                                                                    process_aux_events = process_aux_events,
                                                                    process_weights = process_weights,
                                                                    process_names = process_names,
                                                                    nJ = 2, cuts = cuts)
    sensdict["low_MET_cat_2j"] = low_MET_cat_2j.get_binned_significance(binning = binning, signal_processes = signal_process_names, 
                                                                        background_processes = background_process_names, var_name = "mBB")

    high_MET_cat_2j = CutBasedCategoryFiller.create_high_MET_category(process_events = process_events,
                                                                   process_aux_events = process_aux_events,
                                                                   process_weights = process_weights,
                                                                   process_names = process_names,
                                                                   nJ = 2, cuts = cuts)
    sensdict["high_MET_cat_2j"] = high_MET_cat_2j.get_binned_significance(binning = binning, signal_processes = signal_process_names, 
                                                                        background_processes = background_process_names, var_name = "mBB")

    low_MET_cat_3j = CutBasedCategoryFiller.create_low_MET_category(process_events = process_events,
                                                                    process_aux_events = process_aux_events,
                                                                    process_weights = process_weights,
                                                                    process_names = process_names,
                                                                    nJ = 3, cuts = cuts)
    sensdict["low_MET_cat_3j"] = low_MET_cat_3j.get_binned_significance(binning = binning, signal_processes = signal_process_names, 
                                                                        background_processes = background_process_names, var_name = "mBB")

    high_MET_cat_3j = CutBasedCategoryFiller.create_high_MET_category(process_events = process_events,
                                                                      process_aux_events = process_aux_events,
                                                                      process_weights = process_weights,
                                                                      process_names = process_names,
                                                                      nJ = 3, cuts = cuts)
    sensdict["high_MET_cat_3j"] = high_MET_cat_3j.get_binned_significance(binning = binning, signal_processes = signal_process_names, 
                                                                        background_processes = background_process_names, var_name = "mBB")

    # compute the combined sensitivity
    sensdict["combined"] = np.sqrt(sensdict["low_MET_cat_2j"]**2 + sensdict["high_MET_cat_2j"]**2 + sensdict["low_MET_cat_3j"]**2 + sensdict["high_MET_cat_3j"]**2)
    
    return sensdict

def OptimizeCBASensitivity(infile_path, outdir, do_plots = True):
    test_size = TrainingConfig.test_size # fraction of MC16d events used for the estimation of the expected sensitivity (therefore need to scale up the results by the inverse of this factor)

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
        cur_train, _ = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_traindata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_train) # load the standard classifier input, nuisances and weights
        cur_aux_data = cur_train[TrainingConfig.other_branches].values
        sig_data_train.append(cur_traindata)
        sig_mBB_train.append(cur_nuisdata)
        sig_weights_train.append(cur_weights / (1 - test_size))
        sig_aux_data_train.append(cur_aux_data)

    bkg_data_train = []
    bkg_mBB_train = []
    bkg_weights_train = []
    bkg_aux_data_train = []
    for sample in bkg_data:
        cur_train, _ = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_traindata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_train) # load the standard classifier input, nuisances and weights
        cur_aux_data = cur_train[TrainingConfig.other_branches].values
        bkg_data_train.append(cur_traindata)
        bkg_mBB_train.append(cur_nuisdata)
        bkg_weights_train.append(cur_weights / (1 - test_size))
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

    # the objective function that needs to be minimized
    costfunc = lambda cuts: -EvaluateBinnedSignificance(process_events = data_train, process_aux_events = aux_train, 
                                                       process_weights = weights_train, process_names = samples, 
                                                       signal_process_names = sig_samples, background_process_names = bkg_samples, 
                                                       binning = SR_mBB_binning, cuts = cuts)["combined"]

    costfunc_bayes = lambda MET_cut, dRBB_highMET_cut, dRBB_lowMET_cut: -costfunc({"MET_cut": MET_cut, "dRBB_highMET_cut": dRBB_highMET_cut, "dRBB_lowMET_cut": dRBB_lowMET_cut})

    # the same function, but with a flat array holding the parameters
    # in the order [MET_cut, dRBB_highMET_cut, dRBB_lowMET_cut]
    costfunc_flat = lambda cuts: costfunc({"MET_cut": cuts[0], "dRBB_highMET_cut": cuts[1], "dRBB_lowMET_cut": cuts[2]})

    if do_plots:
        # make some illustrative plots of the sensitivity
        for MET_cut in np.linspace(210.5, 220.5, 11):
            
            dRBB_cut_range = np.linspace(1.0, 1.7, 71)
            res = np.zeros((len(dRBB_cut_range), len(dRBB_cut_range)))
            
            for x, dRBB_highMET_cut in enumerate(dRBB_cut_range):
                for y, dRBB_lowMET_cut in enumerate(dRBB_cut_range):
                    cuts = {"MET_cut": MET_cut, "dRBB_highMET_cut": dRBB_highMET_cut, "dRBB_lowMET_cut": dRBB_lowMET_cut}
                    print("query with {}".format(cuts))
                    res[x,y] = -costfunc(cuts)
                    print(res)

            # generate the plot and store it:
            GenerateHeatMapSensitivityPlot(res, x_ticks = dRBB_cut_range, y_ticks = dRBB_cut_range, title = "MET_cut = {} GeV".format(MET_cut), 
                                           xlabel = "dRBB_highMET_cut", ylabel = "dRBB_lowMET_cut", outfile = os.path.join(outdir, "dRBB_MET_{}.pdf".format(MET_cut)))


    # # perform the optimization:

    # first, try a local optimizer
    # start at the current cut values:
    x0 = [200, 1.2, 1.8]

    # the parameter ranges
    ranges = [[120, 300], [0.5, 3.0], [0.5, 3.0]]
    res_local = minimize(costfunc_flat, x0 = x0, method = 'Nelder-Mead', bounds = ranges, options = {'disp': True})

    # then, try a global search strategy
    ranges_bayes = {"MET_cut": (120, 300), "dRBB_highMET_cut": (0.5, 3.0), "dRBB_lowMET_cut": (0.5, 3.0)}
    gp_params = {'kernel': 1.0 * Matern(length_scale = 0.05, length_scale_bounds = (1e-1, 1e2), nu = 1.5)}
    optimizer = BayesianOptimization(
        f = costfunc_bayes,
        pbounds = ranges_bayes,
        random_state = 1
    )
    optimizer.maximize(init_points = 500, n_iter = 1, acq = 'poi', kappa = 3, **gp_params)

    xi_scheduler = lambda iteration: 0.01 + 0.19 * np.exp(-0.08 * iteration)
    for it in range(300):
        cur_xi = xi_scheduler(it)
        print("using xi = {}".format(cur_xi))
        optimizer.maximize(init_points = 0, n_iter = 1, acq = 'poi', kappa = 3, xi = cur_xi, **gp_params)
    
    # print the results
    print("==============================================")
    print("initial cuts:")
    print("==============================================")
    print("MET_cut = {}".format(x0[0]))
    print("dRBB_highMET_cut = {}".format(x0[1]))
    print("dRBB_lowMET_cut = {}".format(x0[2]))
    print("significance = {} sigma".format(-costfunc_flat(x0)))
    print("==============================================")

    print("==============================================")
    print("optimized cuts (local optimization):")
    print("==============================================")
    print("MET_cut = {}".format(res_local.x[0]))
    print("dRBB_highMET_cut = {}".format(res_local.x[1]))
    print("dRBB_lowMET_cut = {}".format(res_local.x[2]))
    print("significance = {} sigma".format(-costfunc_flat(res_local.x)))
    print("==============================================")

    print("==============================================")
    print("optimized cuts (global optimization):")
    print("==============================================")
    print("MET_cut = {}".format(optimizer.max["params"]["MET_cut"]))
    print("dRBB_highMET_cut = {}".format(optimizer.max["params"]["dRBB_highMET_cut"]))
    print("dRBB_lowMET_cut = {}".format(optimizer.max["params"]["dRBB_lowMET_cut"]))
    print("significance = {} sigma".format(optimizer.max["target"]))
    print("==============================================")
        
if __name__ == "__main__":
    parser = ArgumentParser(description = "optimize the cuts in the CBA for maximum binned significance")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--do_plots", action = "store_const", const = True, default = False)
    args = vars(parser.parse_args())

    outdir = args["outdir"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    OptimizeCBASensitivity(**args)
