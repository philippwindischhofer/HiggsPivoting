import os, argparse, pickle
from argparse import ArgumentParser

from MakeGlobalPlots import MakeGlobalComparisonPlot
from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalSensitivityPlots(model_dirs, plot_dir):
    sensdicts = []

    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "sensdict.pkl"), "rb") as infile:
                sensdict = pickle.load(infile)
                sensdicts.append(sensdict)
        except:
            print("no information found for model '{}'".format(model_dir))

    # generate combined sensitivity plots that show all models
    PerformancePlotter.plot_significance_KS(sensdicts, outfile = os.path.join(plot_dir, "sensitivity_KS.pdf"),
                             model_SRs = ["significance_clf_tight_2J", "significance_clf_loose_2J", "significance_clf_tight_3J", "significance_clf_loose_3J"],
                             model_KSs = ["KS_bkg_class_tight_2J", "KS_bkg_class_loose_2J", "KS_bkg_class_tight_3J", "KS_bkg_class_loose_3J"],
                             reference_SRs = ["significance_low_MET_2J", "significance_high_MET_2J", "significance_low_MET_3J", "significance_high_MET_3J"],
                             reference_KSs = ["KS_bkg_low_MET_2J", "KS_bkg_high_MET_2J", "KS_bkg_low_MET_3J", "KS_bkg_high_MET_3J"])

    # and also using the reduced KS values
    PerformancePlotter.plot_significance_KS(sensdicts, outfile = os.path.join(plot_dir, "sensitivity_KS_red.pdf"),
                             model_SRs = ["significance_clf_tight_2J", "significance_clf_loose_2J", "significance_clf_tight_3J", "significance_clf_loose_3J"],
                             model_KSs = ["KS_bkg_class_tight_2J_red", "KS_bkg_class_loose_2J_red", "KS_bkg_class_tight_3J_red", "KS_bkg_class_loose_3J_red"],
                             reference_SRs = ["significance_low_MET_2J", "significance_high_MET_2J", "significance_low_MET_3J", "significance_high_MET_3J"],
                             reference_KSs = ["KS_bkg_low_MET_2J_red", "KS_bkg_high_MET_2J_red", "KS_bkg_low_MET_3J_red", "KS_bkg_high_MET_3J_red"])

    to_combine = [
        "dist_mBB_class_loose_2J",
        "dist_mBB_class_tight_2J",
        "dist_mBB_class_loose_3J",
        "dist_mBB_class_tight_3J",
        "dist_mBB_sig_class_loose_2J",
        "dist_mBB_sig_class_tight_2J",
        "dist_mBB_sig_class_loose_3J",
        "dist_mBB_sig_class_tight_3J",
        "dist_dRBB_class_loose_2J",
        "dist_dRBB_class_tight_2J",
        "dist_dRBB_class_loose_3J",
        "dist_dRBB_class_tight_3J"
    ]
    overlays = [
        ["dist_mBB_inclusive_2J", "dist_mBB_low_MET_2J"],
        ["dist_mBB_inclusive_2J", "dist_mBB_high_MET_2J"],
        ["dist_mBB_inclusive_3J", "dist_mBB_low_MET_3J"],
        ["dist_mBB_inclusive_3J", "dist_mBB_high_MET_3J"],
        ["dist_mBB_sig_inclusive_2J", "dist_mBB_sig_low_MET_2J"],
        ["dist_mBB_sig_inclusive_2J", "dist_mBB_sig_high_MET_2J"],
        ["dist_mBB_sig_inclusive_3J", "dist_mBB_sig_low_MET_3J"],
        ["dist_mBB_sig_inclusive_3J", "dist_mBB_sig_high_MET_3J"],
        ["dist_dRBB_inclusive_2J", "dist_dRBB_low_MET_2J"],
        ["dist_dRBB_inclusive_2J", "dist_dRBB_high_MET_2J"],
        ["dist_dRBB_inclusive_3J", "dist_dRBB_low_MET_3J"],
        ["dist_dRBB_inclusive_3J", "dist_dRBB_high_MET_3J"]

    ]
    overlay_labels = [
        ["inclusive (nJ = 2)", "cut-based analysis\n (150 GeV < MET < 200 GeV, dRBB < 1.8, nJ = 2)"],
        ["inclusive (nJ = 2)", "cut-based analysis\n (MET > 200 GeV, dRBB < 1.2, nJ = 2)"],
        ["inclusive (nJ = 3)", "cut-based analysis\n (150 GeV < MET < 200 GeV, dRBB < 1.8, nJ = 3)"],
        ["inclusive (nJ = 3)", "cut-based analysis\n (MET > 200 GeV, dRBB < 1.2, nJ = 3)"],
        ["inclusive (nJ = 2)", "cut-based analysis\n (150 GeV < MET < 200 GeV, dRBB < 1.8, nJ = 2)"],
        ["inclusive (nJ = 2)", "cut-based analysis\n (MET > 200 GeV, dRBB < 1.2, nJ = 2)"],
        ["inclusive (nJ = 3)", "cut-based analysis\n (150 GeV < MET < 200 GeV, dRBB < 1.8, nJ = 3)"],
        ["inclusive (nJ = 3)", "cut-based analysis\n (MET > 200 GeV, dRBB < 1.2, nJ = 3)"],
        ["inclusive (nJ = 2)", "cut-based analysis\n (150 GeV < MET < 200 GeV, dRBB < 1.8, nJ = 2)"],
        ["inclusive (nJ = 2)", "cut-based analysis\n (MET > 200 GeV, dRBB < 1.2, nJ = 2)"],
        ["inclusive (nJ = 3)", "cut-based analysis\n (150 GeV < MET < 200 GeV, dRBB < 1.8, nJ = 3)"],
        ["inclusive (nJ = 3)", "cut-based analysis\n (MET > 200 GeV, dRBB < 1.2, nJ = 3)"]
    ]
    xlabels = [
        r'$m_{bb}$ [GeV]',
        r'$m_{bb}$ [GeV]',
        r'$m_{bb}$ [GeV]',
        r'$m_{bb}$ [GeV]',
        r'$m_{bb}$ [GeV]',
        r'$m_{bb}$ [GeV]',
        r'$m_{bb}$ [GeV]',
        r'$m_{bb}$ [GeV]',
        r'dRBB',
        r'dRBB',
        r'dRBB',
        r'dRBB'
    ]
    overlay_colors = ["black", "tomato"]

    for cur, cur_overlay, cur_labels, xlabel in zip(to_combine, overlays, overlay_labels, xlabels):
        # generate plots comparing the mBB shaping in the different signal regions
        MakeGlobalComparisonPlot(model_dirs, outpath = os.path.join(plot_dir + cur + ".pdf"),
                                 source_basename = cur + ".pkl", overlay_paths = [cur_path + ".pkl" for cur_path in cur_overlay],
                                 overlay_labels = cur_labels, overlay_colors = overlay_colors, mode = 'np',
                                 xlabel = xlabel, ylabel = 'a.u.', plot_labels = cur)

def main():
    parser = ArgumentParser(description = "create global sensitivity plots")
    parser.add_argument("--plotdir")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    plot_dir = args["plotdir"]
    model_dirs = args["model_dirs"]

    MakeGlobalSensitivityPlots(model_dirs, plot_dir)

if __name__ == "__main__":
    main()
