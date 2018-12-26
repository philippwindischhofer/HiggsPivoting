import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class PerformancePlotter:

    @staticmethod
    def _II_plot(perfdicts, outpath):
        PerformancePlotter._perfdict_plot(perfdicts, xquant = "logI(f,label)", yquant = "logI(f,nu)", xlabel = r'$\mathrm{log}\,\,I(f, \mathrm{label})$', ylabel = r'$\mathrm{log}\,\,I(f, \nu)$', leglabel = "lambdaleglabel", colorquant = "lambda", outfile = os.path.join(outpath, "II_plot.pdf"))

    @staticmethod
    def _AUC_sig_sq_plot(perfdicts, outpath):
        PerformancePlotter._perfdict_plot(perfdicts, xquant = "ROCAUC", yquant = "sig_sq_diff", xlabel = "AUC", ylabel = "sig_sq_diff_05", leglabel = "lambdaleglabel", colorquant = "lambda", outfile = os.path.join(outpath, "AUC_sig_sq_plot.pdf"))

    @staticmethod
    def _AUC_bkg_sq_plot(perfdicts, outpath):
        PerformancePlotter._perfdict_plot(perfdicts, xquant = "ROCAUC", yquant = "bkg_sq_diff", xlabel = "AUC", ylabel = "bkg_sq_diff_05", leglabel = "lambdaleglabel", colorquant = "lambda", outfile = os.path.join(outpath, "AUC_bkg_sq_plot.pdf"))

    @staticmethod
    def _perfdict_plot(perfdicts, xquant, yquant, xlabel, ylabel, leglabel, colorquant, outfile):
        markerdict = {"MINEClassifierEnvironment": "o", "AdversarialClassifierEnvironment": "v", "data": "s"}
        labeldict = {"MINEClassifierEnvironment": "MINE", "AdversarialClassifierEnvironment": "parametrized posterior", "data": "data"}
        cmap = plt.cm.viridis

        # find the proper normalization of the color map
        colorrange = [float(perfdict[colorquant]) for perfdict in perfdicts if colorquant in perfdict]
        norm = mpl.colors.Normalize(vmin = min(colorrange), vmax = max(colorrange))

        seen_types = []

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for ind, perfdict in enumerate(perfdicts):
            cur_type = perfdict["type"]

            label = labeldict[cur_type]
            color = cmap(norm(float(perfdict[colorquant]))) if colorquant in perfdict else "black"
            marker = markerdict[cur_type]
            if xquant in perfdict and yquant in perfdict:
                if not cur_type in seen_types:
                    seen_types.append(cur_type)

                ax.scatter(perfdict[xquant], perfdict[yquant], color = color, label = label, marker = marker)

        # make legend with the different types that were encountered
        legend_elems = [Line2D([0], [0], marker = markerdict[cur_type], color = 'white', markerfacecolor = "black", label = labeldict[cur_type]) for cur_type in seen_types]
        ax.legend(handles = legend_elems)

        # make colorbar for the range of encountered legended values
        cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.subplots_adjust(right = 0.8)
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap = cmap,
                                       norm = norm,
                                       orientation = 'vertical')
        cb.set_label(r'$\lambda$')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        fig.savefig(outfile)
        plt.close()        

    @staticmethod
    def plot(perfdicts, outpath):
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        PerformancePlotter._II_plot(perfdicts, outpath)
        PerformancePlotter._AUC_sig_sq_plot(perfdicts, outpath)
        PerformancePlotter._AUC_bkg_sq_plot(perfdicts, outpath)
