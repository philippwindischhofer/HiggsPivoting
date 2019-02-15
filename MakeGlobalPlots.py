import os, pickle
from argparse import ArgumentParser

from plotting.PerformancePlotter import PerformancePlotter

def MakeGlobalAUROC_KSPlots(model_dirs, plot_dir):
    perfdicts = []

    # load back the prepared performance metrics
    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "perfdict.pkl"), "rb") as infile:
                perfdict = pickle.load(infile)
            perfdicts.append(perfdict)
        except:
            print("no information found for model '{}'".format(model_dir))

    # generate combined performance plots that compare all the models
    PerformancePlotter.plot_AUROC_KS(perfdicts, outpath = plot_dir, colorquant = "lambda")

def MakeGlobalComparisonPlot(model_dirs, outpath, source_basename, overlay_path = None):
    perfdicts = []
    plot_data = []

    # load the performance dict and also the saved histograms
    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "perfdict.pkl"), "rb") as perfdict_infile, open(os.path.join(model_dir, source_basename), "rb") as plot_infile:
                perfdict = pickle.load(perfdict_infile)
                (n, bins, patches, xlabel, ylabel, plot_labels) = pickle.load(plot_infile)
            perfdicts.append(perfdict)
            plot_data.append((n, bins, patches, xlabel, ylabel, plot_labels))
        except:
            print("no or incomplete information for model '{}'".format(model_dir))

    # load the overlay histogram and plot it as well (always plotted in a neutral color)
    try:
        print("trying to load from {}".format(os.path.join(model_dir, overlay_path)))
        with open(os.path.join(model_dir, overlay_path), "rb") as overlay_infile:
            (bin_contents, edges, _, _, _, _) = pickle.load(overlay_infile)
            low_edges = edges[:-1]
            high_edges = edges[1:]
            centers = 0.5 * (low_edges + high_edges)
            overlays = [(centers, bin_contents, {'color': 'black', 'lw': 1.2, 'label': "inclusive"})]
    except:
        print("problem")
        overlays = []
        
    # generate the combined plot
    PerformancePlotter.combine_hists(perfdicts, plot_data, outpath, colorquant = "lambda", plot_title = os.path.splitext(source_basename)[0], overlays = overlays)

if __name__ == "__main__":
    parser = ArgumentParser(description = "create global comparison plots")
    parser.add_argument("--plotdir")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    plot_dir = args["plotdir"]
    model_dirs = args["model_dirs"]

    # make global plots in the performance / fairness plane
    MakeGlobalAUROC_KSPlots(model_dirs, plot_dir)

    # combine the other plots
    to_combine = [
        "dist_clf_Hbb", "dist_clf_diboson", "dist_clf_ttbar", "dist_clf_Wjets", "dist_clf_singletop", "dist_clf_Zjets",
        "dist_mBB_singletop_50.0", "dist_mBB_Zjets_50.0", "dist_mBB_diboson_50.0", "dist_mBB_ttbar_50.0", "dist_mBB_Hbb_50.0", "dist_mBB_Wjets_50.0",
        "dist_mBB_singletop_25.0", "dist_mBB_Zjets_25.0", "dist_mBB_diboson_25.0", "dist_mBB_ttbar_25.0", "dist_mBB_Hbb_25.0", "dist_mBB_Wjets_25.0",
        "dist_dRBB_singletop_50.0", "dist_dRBB_Zjets_50.0", "dist_dRBB_diboson_50.0", "dist_dRBB_ttbar_50.0", "dist_dRBB_Hbb_50.0", "dist_dRBB_Wjets_50.0",
        "dist_dRBB_singletop_25.0", "dist_dRBB_Zjets_25.0", "dist_dRBB_diboson_25.0", "dist_dRBB_ttbar_25.0", "dist_dRBB_Hbb_25.0", "dist_dRBB_Wjets_25.0",
        "dist_pTB1_singletop_50.0", "dist_pTB1_Zjets_50.0", "dist_pTB1_diboson_50.0", "dist_pTB1_ttbar_50.0", "dist_pTB1_Hbb_50.0", "dist_pTB1_Wjets_50.0",
        "dist_pTB1_singletop_25.0", "dist_pTB1_Zjets_25.0", "dist_pTB1_diboson_25.0", "dist_pTB1_ttbar_25.0", "dist_pTB1_Hbb_25.0", "dist_pTB1_Wjets_25.0",
        "dist_pTB2_singletop_50.0", "dist_pTB2_Zjets_50.0", "dist_pTB2_diboson_50.0", "dist_pTB2_ttbar_50.0", "dist_pTB2_Hbb_50.0", "dist_pTB2_Wjets_50.0",
        "dist_pTB2_singletop_25.0", "dist_pTB2_Zjets_25.0", "dist_pTB2_diboson_25.0", "dist_pTB2_ttbar_25.0", "dist_pTB2_Hbb_25.0", "dist_pTB2_Wjets_25.0"
    ]
    
    overlays = [
        None, None, None, None, None, None,
        "dist_mBB_singletop_100.0", "dist_mBB_Zjets_100.0", "dist_mBB_diboson_100.0", "dist_mBB_ttbar_100.0", "dist_mBB_Hbb_100.0", "dist_mBB_Wjets_100.0",
        "dist_mBB_singletop_100.0", "dist_mBB_Zjets_100.0", "dist_mBB_diboson_100.0", "dist_mBB_ttbar_100.0", "dist_mBB_Hbb_100.0", "dist_mBB_Wjets_100.0",
        "dist_dRBB_singletop_100.0", "dist_dRBB_Zjets_100.0", "dist_dRBB_diboson_100.0", "dist_dRBB_ttbar_100.0", "dist_dRBB_Hbb_100.0", "dist_dRBB_Wjets_100.0",
        "dist_dRBB_singletop_100.0", "dist_dRBB_Zjets_100.0", "dist_dRBB_diboson_100.0", "dist_dRBB_ttbar_100.0", "dist_dRBB_Hbb_100.0", "dist_dRBB_Wjets_100.0"
        "dist_pTB1_singletop_100.0", "dist_pTB1_Zjets_100.0", "dist_pTB1_diboson_100.0", "dist_pTB1_ttbar_100.0", "dist_pTB1_Hbb_100.0", "dist_pTB1_Wjets_100.0",
        "dist_pTB1_singletop_100.0", "dist_pTB1_Zjets_100.0", "dist_pTB1_diboson_100.0", "dist_pTB1_ttbar_100.0", "dist_pTB1_Hbb_100.0", "dist_pTB1_Wjets_100.0",
        "dist_pTB2_singletop_100.0", "dist_pTB2_Zjets_100.0", "dist_pTB2_diboson_100.0", "dist_pTB2_ttbar_100.0", "dist_pTB2_Hbb_100.0", "dist_pTB2_Wjets_100.0",
        "dist_pTB2_singletop_100.0", "dist_pTB2_Zjets_100.0", "dist_pTB2_diboson_100.0", "dist_pTB2_ttbar_100.0", "dist_pTB2_Hbb_100.0", "dist_pTB2_Wjets_100.0"
    ]

    for cur, cur_overlay in zip(to_combine, overlays):
        MakeGlobalComparisonPlot(model_dirs, outpath = os.path.join(plot_dir, cur + ".pdf"), source_basename = cur + ".pkl", overlay_path = cur_overlay + ".pkl" if cur_overlay is not None else None)
