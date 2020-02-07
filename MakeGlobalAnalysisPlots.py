import os, pickle
from argparse import ArgumentParser
import numpy as np

from plotting.PerformancePlotter import PerformancePlotter
from plotting.CategoryPlotter import CategoryPlotter

def MakeGlobalPerformanceFairnessPlots(model_dirs, plotdir):
    dicts = []

    # load back the prepared performance metrics
    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "anadict.pkl"), "rb") as infile:
                anadict = pickle.load(infile)
                if float(anadict["lambda"]) > 1.4:
                    continue
                dicts.append(anadict)
        except:
            print("no information found for model '{}'".format(model_dir))

    dicts = sorted(dicts, key = lambda cur: cur["lambda"])

    PerformancePlotter.plot_significance_fairness_combined(dicts, plotdir, nJ = 2)
    PerformancePlotter.plot_significance_fairness_combined(dicts, plotdir, nJ = 3)

def MakeGlobalAnalysisPlots(outpath, model_dirs, plot_basename, overlay_paths = [], overlay_labels = [], overlay_colors = [], overlay_lss = [], xlabel = "", ylabel = "", plot_label = "", inner_label = [], smoothing = False):
    
    dicts = []
    plot_data = []

    def annotation_epilog(ax):
        ax.text(x = 0.05, y = 0.85, s = "\n".join(inner_label), transform = ax.transAxes,
                horizontalalignment = 'left', verticalalignment = 'bottom')
        ax.set_ylim([0, ax.get_ylim()[1] * 0.5])
    
    # load the plots that are to be mapped over runs
    for model_dir in model_dirs:
        try:
            with open(os.path.join(model_dir, "anadict.pkl"), "rb") as anadict_infile, open(os.path.join(model_dir, plot_basename), "rb") as plot_infile:
                anadict = pickle.load(anadict_infile)
                if float(anadict["lambda"]) > 1.4:
                    continue

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

                overlays.append((centers, n, {'color': overlay_color, 'lw': 2.5, 'label': overlay_label, 'ls': overlay_ls}))
    except:
        overlays = []

    lambdas = [cur["lambda"] for cur in dicts] 
    lambsort = np.argsort(lambdas)
    dicts = [dicts[cur_ind] for cur_ind in lambsort]
    plot_data = [plot_data[cur_ind] for cur_ind in lambsort]

    PerformancePlotter.combine_hists(dicts, plot_data, outpath, colorquant = "lambda", plot_title = "", overlays = overlays, epilog = annotation_epilog, smoothing = smoothing)

def MakeAllGlobalAnalysisPlots(args):
    processes = ["Hbb", "Wjets", "Zjets", "diboson", "ttbar"]
    nJ = [2, 3]
    SRs = ["tight", "loose"]
    CBA_SRs = ["high_MET", "low_MET"]

    dicts = []

    # plot the shaping / performance metrics in all SRs
    MakeGlobalPerformanceFairnessPlots(**args)

    # make shaping plots for all signal regions
    for process in processes:
        for cur_nJ in nJ:
            for cur_SR, cur_CBA_SR in zip(SRs, CBA_SRs):
                filename = "dist_mBB_{}_{}jet_{}.pkl".format(process, cur_nJ, cur_SR)
                overlay_inclusive = os.path.join(args["model_dirs"][0], "dist_mBB_{}_{}jet.pkl".format(process, cur_nJ))
                overlay_CBA_optimized = os.path.join(args["model_dirs"][0], "optimized_dist_mBB_{}_{}jet_{}.pkl".format(process, cur_nJ, cur_CBA_SR))
                overlay_CBA_original = os.path.join(args["model_dirs"][0], "original_dist_mBB_{}_{}jet_{}.pkl".format(process, cur_nJ, cur_CBA_SR))

                overlay_paths = [overlay_inclusive, overlay_CBA_original, overlay_CBA_optimized]
                overlay_labels = ["inclusive", "cut-based analysis", "cut-based analysis\n(optimised)"]
                overlay_colors = ["black", "salmon", "salmon"]
                overlay_lss = ["-", "-", "--"]
                
                outpath = os.path.join(args["plotdir"], "dist_mBB_{}_{}jet_{}.pdf".format(process, cur_nJ, cur_SR))
                outpath_smoothed = os.path.join(args["plotdir"], "dist_mBB_{}_{}jet_{}_smoothed.pdf".format(process, cur_nJ, cur_SR))

                MakeGlobalAnalysisPlots(outpath = outpath, model_dirs = args["model_dirs"], plot_basename = filename, xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.", overlay_paths = overlay_paths, overlay_labels = overlay_labels, overlay_colors = overlay_colors, overlay_lss = overlay_lss, inner_label = [CategoryPlotter.process_labels[process], "{}, {} jet".format(cur_SR, cur_nJ)])
                MakeGlobalAnalysisPlots(outpath = outpath_smoothed, model_dirs = args["model_dirs"], plot_basename = filename, xlabel = r'$m_{bb}$ [GeV]', ylabel = "a.u.", overlay_paths = overlay_paths, overlay_labels = overlay_labels, overlay_colors = overlay_colors, overlay_lss = overlay_lss, inner_label = [CategoryPlotter.process_labels[process], "{}, {} jet".format(cur_SR, cur_nJ)], smoothing = True)
    

if __name__ == "__main__":
    parser = ArgumentParser(description = "create global comparison plots")
    parser.add_argument("--plotdir")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    MakeAllGlobalAnalysisPlots(args)
