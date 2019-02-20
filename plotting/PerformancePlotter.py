import os, re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

class PerformancePlotter:

    @staticmethod
    def plot_significance_KS(sensdicts, outfile):
        # define the signal regions to combine
        model_SRs = ["significance_clf_tight_2J", "significance_clf_loose_2J", "significance_clf_tight_3J", "significance_clf_loose_3J"]
        reference_SRs = ["significance_low_MET_2J", "significance_high_MET_2J", "significance_low_MET_3J", "significance_high_MET_3J"]

        # also define the KS values that go into the plot
        model_KSs = ["KS_bkg_low_MET_2J", "KS_bkg_high_MET_2J", "KS_bkg_low_MET_3J", "KS_bkg_high_MET_3J"]
        reference_KSs = ["KS_bkg_class_tight_2J", "KS_bkg_class_loose_2J", "KS_bkg_class_tight_3J", "KS_bkg_class_loose_3J"]

        # make sure to ge the color normalization correct
        colorquant = "lambda"
        cmap = plt.cm.viridis
        colorrange = [float(sensdict[colorquant]) for sensdict in sensdicts if colorquant in sensdict]
        norm = mpl.colors.Normalize(vmin = min(colorrange), vmax = max(colorrange))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # prepare the individual datasets to plot
        for sensdict in sensdicts:
            # compute the combined significance:
            sigs = [sensdict[cur_sig] for cur_sig in model_SRs]
            combined_sig = np.sqrt(np.sum(np.square(sigs)))
            combined_sigs = np.full(len(sigs), combined_sig)

            # get the contributing KS values for the signal regions:
            KS_vals = [sensdict[cur_KS] for cur_KS in model_KSs]

            color = cmap(norm(float(sensdict[colorquant]))) if colorquant in sensdict else "black"
            ax.plot(combined_sigs, KS_vals, color = color, linestyle = '-', marker = '_')

        # also show the reference model
        sigs = [sensdict[cur_sig] for cur_sig in reference_SRs]
        combined_sig = np.sqrt(np.sum(np.square(sigs)))
        combined_sigs = np.full(len(sigs), combined_sig)

        # get the contributing KS values for the signal regions:
        KS_vals = [sensdict[cur_KS] for cur_KS in reference_KSs]
        ax.plot(combined_sigs, KS_vals, color = "black", linestyle = '-', marker = '_')

        # make colorbar for the range of encountered legended values
        cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.subplots_adjust(right = 0.8)
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap = cmap,
                                       norm = norm,
                                       orientation = 'vertical')
        cb.set_label(r'$\lambda$')

        ax.set_xlabel("expected sensitivity")
        ax.set_ylabel("KS")

        fig.savefig(outfile)
        plt.close()                

    @staticmethod
    def _AUROC_KS_plot(perfdicts, KS_regex, colorquant, outpath):
        typ_perfdict = perfdicts[0]
        available_fields = typ_perfdict.keys()

        KS_fields = filter(KS_regex.match, available_fields)

        get_marker = lambda typestring: 'o'
        get_label = lambda typestring: typestring

        for KS_field in KS_fields:
            PerformancePlotter._perfdict_plot(perfdicts, xquant = "AUROC", yquant = KS_field, xlabel = "AUROC", ylabel = KS_field, 
                                              colorquant = colorquant, markerquant = "adversary_model", markerstyle = get_marker,
                                              markerlabel = get_label, outfile = os.path.join(outpath, "AUROC_" + KS_field + ".pdf"))

    # worker method for flexible plotting: takes as inputs the list of perfdicts created by the ModelEvaluator
    # x/yquant ... quantity that should be printed along x/y
    # x/ylabel ... labels to be used for the x/y axis
    # colorquant ... quantity that controls the color
    # markerquant ... quantity that controls the type of marker that is to be used
    # markerstyle ... lambda of the form marker_style = lambda markerquant: return "marker_style"
    # markerlabel ... lambda of the form legend_entry = lambda markerquant: return "legend_entry"
    @staticmethod
    def _perfdict_plot(perfdicts, xquant, yquant, xlabel, ylabel, colorquant, markerquant, markerstyle, markerlabel, outfile):
        cmap = plt.cm.viridis

        # find the proper normalization of the color map
        colorrange = [float(perfdict[colorquant]) for perfdict in perfdicts if colorquant in perfdict]
        norm = mpl.colors.Normalize(vmin = min(colorrange), vmax = max(colorrange))

        seen_types = []

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which = 'major', color = 'gray', linestyle = '-')
        ax.grid(which = 'minor', color = 'lightgray', linestyle = '--')

        for ind, perfdict in enumerate(perfdicts):
            cur_type = perfdict[markerquant]
            label = markerlabel(cur_type)
            marker = markerstyle(cur_type)
            color = cmap(norm(float(perfdict[colorquant]))) if colorquant in perfdict else "black"

            if xquant in perfdict and yquant in perfdict:
                if not cur_type in seen_types:
                    seen_types.append(cur_type)

                ax.scatter(perfdict[xquant], perfdict[yquant], color = color, label = label, marker = marker)

        # make legend with the different types that were encountered
        legend_elems = [Line2D([0], [0], marker = markerstyle(cur_type), color = 'white', markerfacecolor = "black", label = markerlabel(cur_type)) for cur_type in seen_types]
        ax.legend(handles = legend_elems)

        # make colorbar for the range of encountered legended values
        cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.subplots_adjust(right = 0.8)
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap = cmap,
                                       norm = norm,
                                       orientation = 'vertical')
        cb.set_label(r'$\lambda$')

        # set limits
        ax.set_xlim([0.5, 1.0])
        ax.set_ylim([0.0, 1.0])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        fig.savefig(outfile)
        plt.close()        

    @staticmethod
    def plot_AUROC_KS(perfdicts, outpath, colorquant = "lambda"):
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        PerformancePlotter._AUROC_KS_plot(perfdicts, re.compile("KS_50_.*"), colorquant, outpath)
        PerformancePlotter._AUROC_KS_plot(perfdicts, re.compile("KS_25_.*"), colorquant, outpath)

    # combine the passed plots and save them
    @staticmethod
    def combine_hists(perfdicts, hist_data, outpath, colorquant, plot_title, overlays = []):
        cmap = plt.cm.viridis

        # find the proper normalization of the color map
        colorrange = [float(perfdict[colorquant]) for perfdict in perfdicts if colorquant in perfdict]
        norm = mpl.colors.Normalize(vmin = min(colorrange), vmax = max(colorrange))

        if len(perfdicts) != len(hist_data):
            raise Exception("need to get the same number of perfdicts and plots")

        bin_values = []
        bin_centers = []
        colors = []

        for perfdict, cur_hist in zip(perfdicts, hist_data):
            cur_bin_values = cur_hist[0]
            edges = cur_hist[1]
            xlabel = cur_hist[3]
            ylabel = cur_hist[4]

            color = cmap(norm(float(perfdict[colorquant]))) if colorquant in perfdict else "black"
            colors.append(color)

            low_edges = edges[:-1]
            high_edges = edges[1:]
            cur_centers = 0.5 * (low_edges + high_edges)

            bin_centers.append(cur_centers)
            bin_values.append(cur_bin_values)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # plot the combined histograms
        for cur_bin_centers, cur_bin_values, cur_color in zip(bin_centers, bin_values, colors):
            ax.plot(cur_bin_centers, cur_bin_values, color = cur_color, linewidth = 0.1)
        
        # plot the overlays
        for (x, y, opts) in overlays:
            ax.plot(x, y, **opts)
            ax.legend()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(plot_title)

        # make colorbar for the range of encountered legended values
        cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.subplots_adjust(right = 0.8)
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap = cmap,
                                       norm = norm,
                                       orientation = 'vertical')
        cb.set_label(r'$\lambda$')

        fig.savefig(outpath)
        plt.close()
