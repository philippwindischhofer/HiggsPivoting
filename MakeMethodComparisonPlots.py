import os, pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from plotting.CategoryPlotter import CategoryPlotter
from plotting.PerformancePlotter import PerformancePlotter
from MakeMIEvolutionPlot import _load_metadata

def MakeMethodComparisonPlots(outpath, model_dirs, plot_basename, overlay_paths = [], overlay_labels = [], overlay_colors = [], overlay_lss = [], xlabel = "", ylabel = "", plot_label = "", inner_label = []):

    adversary_label_library = {"MINEAdversary": "MIND", "DisCoAdversary": "DisCo", "GMMAdversary": "EMAX"}
    color_library = {"MINEAdversary": plt.cm.Blues(0.7), "DisCoAdversary": plt.cm.Oranges(0.7), "GMMAdversary": plt.cm.Greens(0.7)}    
    
    dicts = []
    colors = []
    labels = []

    plot_data = []

    def annotation_epilog(ax):
        ax.text(x = 0.05, y = 0.85 - 0.05 * (len(inner_label) - 2), s = "\n".join(inner_label), transform = ax.transAxes,
                horizontalalignment = 'left', verticalalignment = 'bottom')
        ax.set_ylim([0, ax.get_ylim()[1] * 0.5])

    # load the plots that are to be mapped over runs
    for model_dir in model_dirs:

        adv_model = _load_metadata(os.path.join(model_dir, "meta.conf"), "AdversarialEnvironment")["adversary_model"]
        lambda_val = _load_metadata(os.path.join(model_dir, "meta.conf"), "AdversarialEnvironment")["lambda"]
        adversary_label = adversary_label_library[adv_model]
        color = color_library[adv_model]
        colors.append(color)
        labels.append(adversary_label + r" ($\lambda_{{\mathrm{{{}}}}} = {}$)".format(adversary_label, lambda_val))

        try:
            with open(os.path.join(model_dir, "anadict.pkl"), "rb") as anadict_infile, open(os.path.join(model_dir, plot_basename), "rb") as plot_infile:
                anadict = pickle.load(anadict_infile)

                (n, bins, var_name) = pickle.load(plot_infile)

                dicts.append(anadict)
                plot_data.append((n, bins, xlabel, ylabel, plot_label))
        except:
            print("no or incomplete information for model '{}'".format(model_dir))

    # load the overlay histograms that are to be plotted on top
    try:
        overlays = []

        for overlay_path, overlay_label, overlay_color, overlay_ls in zip(overlay_paths, overlay_labels, overlay_colors, overlay_lss):
            print("trying to load from {}".format(overlay_path))

            with open(overlay_path, "rb") as overlay_infile:
                (n, bins, var_name) = pickle.load(overlay_infile)
                
                low_edges = bins[:-1]
                high_edges = bins[1:]
                centers = 0.5 * (low_edges + high_edges)

                overlays.append((centers, n, {'color': overlay_color, 'lw': 2.0, 'label': overlay_label, 'ls': overlay_ls}))
    except:
        overlays = []

    PerformancePlotter.combine_hists_simple(plot_data, outpath, colors, labels, overlays = overlays, epilog = annotation_epilog, xlabel = r"$m_{bb}$ [GeV]", ylabel = "a.u.")

def MakeAllMethodComparisonPlots(plotdir, workdirs):

    processes = ["Hbb", "Wjets", "Zjets", "diboson", "ttbar", "bkg"]
    nJ = [2, 3]
    SRs = ["tight", "loose"]
    CBA_SRs = ["high_MET", "low_MET"]

    for process in processes:
        for cur_nJ in nJ:
            for cur_SR, cur_CBA_SR in zip(SRs, CBA_SRs):
                filename = "dist_mBB_{}_{}jet_{}.pkl".format(process, cur_nJ, cur_SR)
                overlay_inclusive = os.path.join(workdirs[0], "dist_mBB_{}_{}jet.pkl".format(process, cur_nJ))
                overlay_CBA_optimized = os.path.join(workdirs[0], "optimized_dist_mBB_{}_{}jet_{}.pkl".format(process, cur_nJ, cur_CBA_SR))
                overlay_CBA_original = os.path.join(workdirs[0], "original_dist_mBB_{}_{}jet_{}.pkl".format(process, cur_nJ, cur_CBA_SR))

                overlay_paths = [overlay_inclusive, overlay_CBA_original, overlay_CBA_optimized]
                overlay_labels = ["inclusive", "cut-based analysis", "cut-based analysis\n(optimised)"]
                overlay_colors = ["black", "darkgrey", "darkgrey"]
                overlay_lss = ["-", "-", "--"]
                
                outpath = os.path.join(args["plotdir"], "dist_mBB_{}_{}jet_{}.pdf".format(process, cur_nJ, cur_SR))

                MakeMethodComparisonPlots(outpath, model_dirs = workdirs, plot_basename = filename, overlay_paths = overlay_paths, overlay_labels = overlay_labels, overlay_lss = overlay_lss, overlay_colors = overlay_colors, inner_label = [CategoryPlotter.process_labels[process], "{}, {} jet".format(cur_SR, cur_nJ)])

if __name__ == "__main__":
    parser = ArgumentParser(description = "make comparison plots across different methods")
    parser.add_argument("--plotdir", action = "store")
    parser.add_argument("--workdirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    MakeAllMethodComparisonPlots(**args)
