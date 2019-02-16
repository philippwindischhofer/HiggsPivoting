import os
from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split

from models.AdversarialEnvironment import AdversarialEnvironment
from analysis.Category import Category
from analysis.CutBasedCategoryFiller import CutBasedCategoryFiller
from analysis.ClassifierBasedCategoryFiller import ClassifierBasedCategoryFiller
from plotting.CategoryPlotter import CategoryPlotter
from DatasetExtractor import TrainNuisAuxSplit

def main():
    parser = ArgumentParser(description = "evaluate expected sensitivity")
    parser.add_argument("--data", action = "store", dest = "infile_path")
    parser.add_argument("--model_dir", action = "store", dest = "model_dir")
    parser.add_argument("--plotdir", action = "store", dest = "plotdir")
    args = vars(parser.parse_args())

    infile_path = args["infile_path"]
    model_dir = args["model_dir"]
    plotdir = args["plotdir"]

    test_size = 0.2 # fraction of MC16d events used for the estimation of the expected sensitivity (therefore need to scale up the results by the inverse of this factor)
    lumi = 50 # total integrated luminosity for MC16d

    # read the test dataset, which will be used to get the expected sensitivity of the analysis
    sig_samples = ["Hbb"]
    bkg_samples = ["ttbar", "Zjets", "Wjets", "diboson", "singletop"]

    print("loading data ...")
    sig_data = [pd.read_hdf(infile_path, key = sig_sample) for sig_sample in sig_samples]
    bkg_data = [pd.read_hdf(infile_path, key = bkg_sample) for bkg_sample in bkg_samples]

    sig_data_test = []
    sig_mBB_test = []
    sig_weights_test = []
    for sample in sig_data:
        _, cur_test = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights
        sig_data_test.append(cur_testdata)
        sig_mBB_test.append(cur_nuisdata)
        sig_weights_test.append(cur_weights / test_size)

    bkg_data_test = []
    bkg_mBB_test = []
    bkg_weights_test = []
    for sample in bkg_data:
        _, cur_test = train_test_split(sample, test_size = test_size, shuffle = True, random_state = 12345)
        cur_testdata, cur_nuisdata, cur_weights = TrainNuisAuxSplit(cur_test) # load the standard classifier input, nuisances and weights
        bkg_data_test.append(cur_testdata)
        bkg_mBB_test.append(cur_nuisdata)
        bkg_weights_test.append(cur_weights / test_size)

    # also prepare the total, concatenated versions
    data_test = sig_data_test + bkg_data_test
    weights_test = sig_weights_test + bkg_weights_test
    samples = sig_samples + bkg_samples

    # first, plot the total event content (i.e. corresponding to an "inclusive" event category)
    inclusive = Category("inclusive")
    for events, weights, process in zip(sig_data_test + bkg_data_test, sig_weights_test + bkg_weights_test, sig_samples + bkg_samples):
        inclusive.add_events(events = events, weights = weights, process = process)

    # show the inclusive event content
    CategoryPlotter.plot_category_composition(inclusive, outpath = os.path.join(plotdir, "dist_mBB_inclusive.pdf"), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', plotlabel = ["MC16d only", "inclusive"])

    # create the event categories for the simple cut-based analysis
    low_MET_cat = CutBasedCategoryFiller.create_low_MET_category(process_events = data_test,
                                                                 process_weights = weights_test,
                                                                 process_names = samples)
    high_MET_cat = CutBasedCategoryFiller.create_high_MET_category(process_events = data_test,
                                                                   process_weights = weights_test,
                                                                   process_names = samples)

    # also show the distributions in these two categories
    CategoryPlotter.plot_category_composition(low_MET_cat, outpath = os.path.join(plotdir, "dist_mBB_low_MET.pdf"), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                              plotlabel = ["MC16d only", r'150 GeV < MET < 200 GeV', "dRBB < 1.8"], args = {"range": (25, 200), "bins": 25})
    CategoryPlotter.plot_category_composition(high_MET_cat, outpath = os.path.join(plotdir, "dist_mBB_high_MET.pdf"), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                              plotlabel = ["MC16d only", "MET > 200 GeV", "dRBB < 1.2"], args = {"range": (25, 200), "bins": 25})

    # load the classifier model and also fill two classifier-based categories
    env = AdversarialEnvironment.from_file(model_dir)
    class_cat = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                         process_events = data_test,
                                                                         process_weights = weights_test,
                                                                         process_names = samples,
                                                                         signal_events = sig_data_test,
                                                                         signal_weights = sig_weights_test,
                                                                         classifier_sigeff_range = (0.25, 0.0))
    CategoryPlotter.plot_category_composition(class_cat, outpath = os.path.join(plotdir, "dist_mBB_class_tight.pdf"), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                              plotlabel = ["MC16d only", class_cat.name], args = {"range": (25, 200), "bins": 25})

    class_cat = ClassifierBasedCategoryFiller.create_classifier_category(env, 
                                                                         process_events = data_test,
                                                                         process_weights = weights_test,
                                                                         process_names = samples,
                                                                         signal_events = sig_data_test,
                                                                         signal_weights = sig_weights_test,
                                                                         classifier_sigeff_range = (0.75, 0.25))
    CategoryPlotter.plot_category_composition(class_cat, outpath = os.path.join(plotdir, "dist_mBB_class_loose.pdf"), var = "mBB", xlabel = r'$m_{bb}$ [GeV]', 
                                              plotlabel = ["MC16d only", class_cat.name], args = {"range": (25, 200), "bins": 25})

if __name__ == "__main__":
    main()
