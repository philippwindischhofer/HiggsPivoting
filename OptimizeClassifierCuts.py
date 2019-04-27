import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from argparse import ArgumentParser
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process.kernels import Matern

from analysis.Category import Category
from analysis.ClassifierBasedCategoryFiller import ClassifierBasedCategoryFiller
from models.AdversarialEnvironment import AdversarialEnvironment
from base.Configs import TrainingConfig
from DatasetExtractor import TrainNuisAuxSplit

def EvaluateBinnedSignificance(sig_events, sig_aux_events, sig_weights, sig_process_names, sig_preds,
                               bkg_events, bkg_aux_events, bkg_weights, bkg_process_names, bkg_preds, env, binning, cuts):

    sensdict = {}

    clf_cut_tight = ClassifierBasedCategoryFiller._sigeff_to_score(env, sig_events, sig_weights, cuts["tight"])
    clf_cut_loose = ClassifierBasedCategoryFiller._sigeff_to_score(env, sig_events, sig_weights, cuts["loose"])

    print("converted tight sigeff cut = {} to classifier cut = {}".format(cuts["tight"], clf_cut_tight))
    print("converted loose sigeff cut = {} to classifier cut = {}".format(cuts["loose"], clf_cut_loose))

    # fill the four event categories (classifier tight & loose, each split into 2 jet and 3 jet)
    class_cat_tight_2J = ClassifierBasedCategoryFiller.create_classifier_category(env,
                                                                                  process_events = sig_events + bkg_events,
                                                                                  process_aux_events = sig_aux_events + bkg_aux_events,
                                                                                  process_weights = sig_weights + bkg_weights,
                                                                                  process_names = sig_process_names + bkg_process_names,
                                                                                  process_preds = sig_preds + bkg_preds,
                                                                                  signal_events = sig_events,
                                                                                  signal_weights = sig_weights,
                                                                                  classifier_sigeff_range = (clf_cut_tight, 1.0),
                                                                                  interpret_as_sigeff = False,
                                                                                  nJ = 2)
    sensdict["clf_tight_2j"] = class_cat_tight_2J.get_binned_significance(binning = binning, signal_processes = sig_process_names, 
                                                                          background_processes = bkg_process_names, var_name = "mBB")

    class_cat_loose_2J = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                  process_events = sig_events + bkg_events,
                                                                                  process_aux_events = sig_aux_events + bkg_aux_events,
                                                                                  process_weights = sig_weights + bkg_weights,
                                                                                  process_names = sig_process_names + bkg_process_names,
                                                                                  process_preds = sig_preds + bkg_preds,
                                                                                  signal_events = sig_events,
                                                                                  signal_weights = sig_weights,
                                                                                  classifier_sigeff_range = (clf_cut_loose, clf_cut_tight),
                                                                                  interpret_as_sigeff = False,
                                                                                  nJ = 2)
    sensdict["clf_loose_2j"] = class_cat_loose_2J.get_binned_significance(binning = binning, signal_processes = sig_process_names, 
                                                                          background_processes = bkg_process_names, var_name = "mBB")

    class_cat_tight_3J = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                  process_events = sig_events + bkg_events,
                                                                                  process_aux_events = sig_aux_events + bkg_aux_events,
                                                                                  process_weights = sig_weights + bkg_weights,
                                                                                  process_names = sig_process_names + bkg_process_names,
                                                                                  process_preds = sig_preds + bkg_preds,
                                                                                  signal_events = sig_events,
                                                                                  signal_weights = sig_weights,
                                                                                  classifier_sigeff_range = (clf_cut_tight, 1.0),
                                                                                  interpret_as_sigeff = False,
                                                                                  nJ = 3)
    sensdict["clf_tight_3j"] = class_cat_tight_3J.get_binned_significance(binning = binning, signal_processes = sig_process_names, 
                                                                          background_processes = bkg_process_names, var_name = "mBB")

    class_cat_loose_3J = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                                  process_events = sig_events + bkg_events,
                                                                                  process_aux_events = sig_aux_events + bkg_aux_events,
                                                                                  process_weights = sig_weights + bkg_weights,
                                                                                  process_names = sig_process_names + bkg_process_names,
                                                                                  process_preds = sig_preds + bkg_preds,
                                                                                  signal_events = sig_events,
                                                                                  signal_weights = sig_weights,
                                                                                  classifier_sigeff_range = (clf_cut_loose, clf_cut_tight),
                                                                                  interpret_as_sigeff = False,
                                                                                  nJ = 3)
    sensdict["clf_loose_3j"] = class_cat_loose_3J.get_binned_significance(binning = binning, signal_processes = sig_process_names, 
                                                                          background_processes = bkg_process_names, var_name = "mBB")

    # compute the combined binned sensitivity
    sensdict["combined"] = np.sqrt(sensdict["clf_tight_2j"] ** 2 + sensdict["clf_loose_2j"] ** 2 + sensdict["clf_tight_3j"] ** 2 + sensdict["clf_loose_3j"] ** 2)

    return sensdict

def OptimizeClassifierCuts(infile_path, model_dir, out_dir, do_local = False, do_global = False):
    sig_samples = TrainingConfig.sig_samples
    bkg_samples = TrainingConfig.bkg_samples
    test_size = TrainingConfig.test_size

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

    # apply the classifier up-front to save time later
    sig_preds = [env.predict(data = cur_events)[:,1] for cur_events in sig_data_test]
    bkg_preds = [env.predict(data = cur_events)[:,1] for cur_events in bkg_data_test]

    # define the SR binning for mBB
    SR_low = 30
    SR_up = 210
    SR_binwidth = 10
    SR_mBB_binning = np.linspace(SR_low, SR_up, num = int((SR_up - SR_low) / SR_binwidth), endpoint = True)

    # this is the target function that should be optimized
    costfunc = lambda cuts: -EvaluateBinnedSignificance(sig_events = sig_data_test, sig_aux_events = sig_aux_data_test, sig_weights = sig_weights_test, sig_process_names = sig_samples, sig_preds = sig_preds,
                                                        bkg_events = bkg_data_test, bkg_aux_events = bkg_aux_data_test, bkg_weights = bkg_weights_test, bkg_process_names = bkg_samples, bkg_preds = bkg_preds,
                                                        env = env, binning = SR_mBB_binning, cuts = cuts)["combined"]

    # first argument directly gives the loose cut, second argument gives the tight cut in units of the loose cut
    costfunc_flat = lambda cuts: costfunc({"loose": cuts[0], "tight": cuts[0] * cuts[1]})
    
    # start at some reasonable cut values: tight_cut = 0.3, loose_cut = 0.8
    x0 = [0.8, 0.3 / 0.8]
    cuts_initial = {"loose": x0[0], "tight": x0[0] * x0[1]}
    cuts_initial["sig"] = -costfunc(cuts_initial)

    ranges = [[0.0, 1.0], [0.0, 1.0]]

    if do_local:
        res_local = minimize(costfunc_flat, x0 = x0, method = 'Nelder-Mead', bounds = ranges, options = {'disp': True})
        cutopt_local = {"loose": res_local.x[0], "tight": res_local.x[0] * res_local.x[1]}
        cutopt_local["sig"] = -costfunc_flat(res_local.x)
    else:
        cutopt_local = cuts_initial

    # then, try a global (Bayesian) optimization
    costfunc_bayes = lambda c0, c1: -costfunc_flat((c0, c1))

    if do_global:
        ranges_bayes = {"c0": (0.0, 1.0), "c1": (0.0, 1.0)}
        gp_params = {'kernel': 1.0 * Matern(length_scale = 0.05, length_scale_bounds = (1e-1, 1e2), nu = 1.5)}
        optimizer = BayesianOptimization(
            f = costfunc_bayes,
            pbounds = ranges_bayes,
            random_state = 1
        )
        optimizer.maximize(init_points = 20, n_iter = 1, acq = 'poi', kappa = 3, **gp_params)

        xi_scheduler = lambda iteration: 0.01 + 0.19 * np.exp(-0.03 * iteration)
        for it in range(40):
            cur_xi = xi_scheduler(it)
            print("using xi = {}".format(cur_xi))
            optimizer.maximize(init_points = 0, n_iter = 1, acq = 'poi', kappa = 3, xi = cur_xi, **gp_params)

        cutopt_global = {"loose": optimizer.max["params"]["c0"], "tight": optimizer.max["params"]["c0"] * optimizer.max["params"]["c1"]}
        cutopt_global["sig"] = -costfunc(cutopt_global)
    else:
        cutopt_global = cuts_initial
    
    print("==============================================")
    print("initial cuts:")
    print("==============================================")
    print("loose cut = {}".format(cuts_initial["loose"]))
    print("tight cut = {}".format(cuts_initial["tight"]))
    print("significance = {} sigma".format(cuts_initial["sig"]))
    print("==============================================")

    print("==============================================")
    print("optimized cuts (local optimization):")
    print("==============================================")
    print("loose cut = {}".format(cutopt_local["loose"]))
    print("tight cut = {}".format(cutopt_local["tight"]))
    print("significance = {} sigma".format(cutopt_local["sig"]))
    print("==============================================")

    print("==============================================")
    print("optimized cuts (global optimization):")
    print("==============================================")
    print("loose cut = {}".format(cutopt_global["loose"]))
    print("tight_cut = {}".format(cutopt_global["tight"]))
    print("significance = {} sigma".format(cutopt_global["sig"]))
    print("==============================================")

    # save the best result
    results = [cuts_initial, cutopt_local, cutopt_global]
    results = sorted(results, key = lambda cur: cur["sig"], reverse = True)
    cutdict = results[0]

    with open(os.path.join(out_dir, "cutdict.pkl"), "wb") as outfile:
        print("saving the following cuts:")
        print(cutdict)
        pickle.dump(cutdict, outfile)

if __name__ == "__main__":
    parser = ArgumentParser(description = "optimize cuts on classifier output")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--model_dir", action = "store", dest = "model_dir")
    parser.add_argument("--outdir", action = "store", dest = "out_dir")
    parser.add_argument("--do_local", action = "store_const", const = True, dest = "do_local")
    parser.add_argument("--do_global", action = "store_const", const = True, dest = "do_global")
    args = vars(parser.parse_args())

    OptimizeClassifierCuts(**args)
