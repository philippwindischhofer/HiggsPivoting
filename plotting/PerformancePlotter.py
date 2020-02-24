import os, re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib.lines import Line2D
import numpy as np

class PerformancePlotter:

    @staticmethod
    def plot_asimov_significance_category_sweep_comparison(hypodicts, catdicts, outfile, xlabel = "# categories", ylabel = r'Asimov significance [$\sigma_A$]',
                                                           asimov_sig_name = "asimov_sig_background_fixed", lambdas_to_plot = [0.0, 0.25, 0.75]):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # load all the fit results
        asimov_sigs = {}
        category_numbers = []
        lambda_values = []
        for hypodict, catdict in zip(hypodicts, catdicts):
            
            cur_asimov_sig = hypodict[asimov_sig_name]
            cur_number_categories = catdict["num_categories"]
            cur_lambda = float(catdict["lambda"])

            category_numbers.append(cur_number_categories)
            lambda_values.append(cur_lambda)

            if not (cur_lambda, cur_number_categories) in asimov_sigs:
                asimov_sigs[(cur_lambda, cur_number_categories)] = []
            asimov_sigs[(cur_lambda, cur_number_categories)].append(cur_asimov_sig)
        
        category_numbers = sorted(list(set(category_numbers)))
        lambda_values = sorted(list(set(lambda_values)))

        # compute the mean values and standard deviations over trainings and plot them
        for cur_lambda_value in lambda_values:
            if min([abs(cur_lambda_value - test_lambda) for test_lambda in lambdas_to_plot]) < 1e-4:
                asimov_sigs_means = [np.mean(asimov_sigs[(cur_lambda_value, cur_category_number)]) for cur_category_number in category_numbers]
                asimov_sigs_errors = [np.std(asimov_sigs[(cur_lambda_value, cur_category_number)]) for cur_category_number in category_numbers]

                ax.errorbar(category_numbers, asimov_sigs_means, yerr = asimov_sigs_errors, marker = 'o', label = r'$\lambda = {}$'.format(cur_lambda_value), fmt = 'o')

        ax.legend()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(asimov_sig_name)
        
        fig.savefig(outfile)
        plt.close()                

    # expect CBA_overlays in the form [{label, ls, dict}]
    @staticmethod
    def plot_asimov_significance_comparison(hypodicts_runs, sensdicts_runs, outdir, labels, colors, xlabel = r'$\lambda$', ylabel = r'Asimov significance [$\sigma_A$]',
                                            model_SRs = ["significance_clf_tight_2J", "significance_clf_loose_2J", "significance_clf_tight_3J", "significance_clf_loose_3J"], plotlabel = []):

        assert len(hypodicts_runs) == len(sensdicts_runs) # make sure are given corresponding information

        def assemble_plotdata(hypodicts, sensdicts):
            asimov_sigs_ncat_background_fixed = {}
            asimov_sigs_ncat_background_floating = {}

            # prepare the individual datasets to plot
            for sensdict, hypodict in zip(sensdicts, hypodicts):
                asimov_sig_ncat_background_fixed = hypodict["asimov_sig_ncat_background_fixed"]
                asimov_sig_ncat_background_floating = hypodict["asimov_sig_ncat_background_floating"]
                                
                cur_lambda = float(sensdict["lambda"])
                
                if cur_lambda not in asimov_sigs_ncat_background_fixed:
                    asimov_sigs_ncat_background_fixed[cur_lambda] = []
                asimov_sigs_ncat_background_fixed[cur_lambda].append(asimov_sig_ncat_background_fixed)

                if cur_lambda not in asimov_sigs_ncat_background_floating:
                    asimov_sigs_ncat_background_floating[cur_lambda] = []
                asimov_sigs_ncat_background_floating[cur_lambda].append(asimov_sig_ncat_background_floating)

            # compute the size of the error bars (show the mean as central value, and the standard deviation as uncertainty measure)
            lambdas = np.array(list(asimov_sigs_ncat_background_fixed.keys()))
            sortind = np.argsort(lambdas)
            lambdas = lambdas[sortind]

            asimov_sigs_ncat_background_fixed_mean = np.array([np.mean(cur) for cur in asimov_sigs_ncat_background_fixed.values()])[sortind]
            asimov_sigs_ncat_background_fixed_std = np.array([np.std(cur) for cur in asimov_sigs_ncat_background_fixed.values()])[sortind]
            asimov_sigs_ncat_background_fixed_max = np.array([np.max(cur) for cur in asimov_sigs_ncat_background_fixed.values()])[sortind]
            asimov_sigs_ncat_background_fixed_min = np.array([np.min(cur) for cur in asimov_sigs_ncat_background_fixed.values()])[sortind]

            asimov_sigs_ncat_background_floating_mean = np.array([np.mean(cur) for cur in asimov_sigs_ncat_background_floating.values()])[sortind]
            asimov_sigs_ncat_background_floating_std = np.array([np.std(cur) for cur in asimov_sigs_ncat_background_floating.values()])[sortind]
            asimov_sigs_ncat_background_floating_max = np.array([np.max(cur) for cur in asimov_sigs_ncat_background_floating.values()])[sortind]
            asimov_sigs_ncat_background_floating_min = np.array([np.min(cur) for cur in asimov_sigs_ncat_background_floating.values()])[sortind]

            unc_up_background_floating = asimov_sigs_ncat_background_floating_max - asimov_sigs_ncat_background_floating_mean
            unc_down_background_floating = asimov_sigs_ncat_background_floating_min - asimov_sigs_ncat_background_floating_mean

            unc_up_background_fixed = asimov_sigs_ncat_background_fixed_max - asimov_sigs_ncat_background_fixed_mean
            unc_down_background_fixed = asimov_sigs_ncat_background_fixed_min - asimov_sigs_ncat_background_fixed_mean

            return lambdas, asimov_sigs_ncat_background_floating_mean, unc_up_background_floating, unc_down_background_floating, asimov_sigs_ncat_background_fixed_mean, unc_up_background_fixed, unc_down_background_fixed
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        lambdas = []
        asimov_sigs_ncat_background_floating_means = []
        uncs_up_background_floating = []
        uncs_down_background_floating = []
        asimov_sigs_ncat_background_fixed_means = []
        uncs_up_background_fixed = []
        uncs_down_background_fixed = []

        for hypodicts, sensdicts in zip(hypodicts_runs, sensdicts_runs):
            cur_lambdas, cur_asimov_sigs_ncat_background_floating_mean, cur_unc_up_background_floating, cur_unc_down_background_floating, cur_asimov_sigs_ncat_background_fixed_mean, cur_unc_up_background_fixed, cur_unc_down_background_fixed = assemble_plotdata(hypodicts, sensdicts)

            lambdas.append(cur_lambdas)
            asimov_sigs_ncat_background_floating_means.append(cur_asimov_sigs_ncat_background_floating_mean)
            uncs_up_background_floating.append(cur_unc_up_background_floating)
            uncs_down_background_floating.append(cur_unc_up_background_floating)
            asimov_sigs_ncat_background_fixed_means.append(cur_asimov_sigs_ncat_background_fixed_mean)
            uncs_up_background_fixed.append(cur_unc_up_background_fixed)            
            uncs_down_background_fixed.append(cur_unc_down_background_fixed)


        def bkg_floating_epilog(ax):
            ax.axhline(y = hypodicts_runs[0][0]["optimized_asimov_sig_high_low_MET_background_floating"], xmin = 0.0, xmax = 1.0, 
                       color = "salmon", linestyle = "--", label = "cut-based analysis (optimised)", zorder = 1)
            ax.axhline(y = hypodicts_runs[0][0]["original_asimov_sig_high_low_MET_background_floating"], xmin = 0.0, xmax = 1.0, 
                       color = "salmon", linestyle = "-", label = "cut-based analysis", zorder = 1)
            
        def bkg_fixed_epilog(ax):
            ax.axhline(y = hypodicts_runs[0][0]["optimized_asimov_sig_high_low_MET_background_fixed"], xmin = 0.0, xmax = 1.0, 
                       color = "indianred", linestyle = ":", label = "cut-based analysis (optimised)")
            ax.axhline(y = hypodicts_runs[0][0]["original_asimov_sig_high_low_MET_background_fixed"], xmin = 0.0, xmax = 1.0, 
                       color = "indianred", linestyle = "-", label = "cut-based analysis")
            
        PerformancePlotter._uncertainty_plot(lambdas, asimov_sigs_ncat_background_floating_means, uncs_up = uncs_up_background_floating, uncs_down = uncs_down_background_floating, 
                                             labels = labels, outfile = os.path.join(outdir, "asimov_significance_background_floating.pdf"), xlabel = xlabel, ylabel = ylabel, colors = colors, title = "",
                                             epilog = bkg_floating_epilog,
                                             plotlabel = plotlabel)

        PerformancePlotter._uncertainty_plot(lambdas, asimov_sigs_ncat_background_fixed_means, uncs_up = uncs_up_background_fixed, uncs_down = uncs_down_background_fixed, 
                                             labels = labels, outfile = os.path.join(outdir, "asimov_significance_background_fixed.pdf"), xlabel = xlabel, ylabel = ylabel, colors = colors, title = "background fixed",
                                             epilog = bkg_fixed_epilog,
                                             plotlabel = plotlabel)

    @staticmethod
    def _uncertainty_plot(xs, ys, uncs_up, uncs_down, labels, outfile, xlabel, ylabel, colors, show_legend = True, epilog = None, title = "", plotlabel = []):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if epilog is not None:
            epilog(ax)

        for x, y, unc_up, unc_down, label, color in zip(xs, ys, uncs_up, uncs_down, labels, colors):
            # first, plot the central value
            ax.plot(x, y, marker = 'o', label = label, color = color)
            ax.fill_between(x, y + unc_down, y + unc_up, color = color, alpha = 0.6, zorder = 2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.margins(x = 0.0, y = 0.1)

        if plotlabel:
            text = "\n".join(plotlabel)
            plt.text(0.4, 0.95, text, verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

        if show_legend:
            leg = ax.legend(loc = 'lower right')
            leg.get_frame().set_linewidth(0.0)

        fig.savefig(outfile)
        plt.close()

    @staticmethod
    def _simple_plot(xs, ys, colors, linestyles, linestyle_labels, color_labels, outfile, xlabel, ylabel):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for x, y, color, linestyle in zip(xs, ys, colors, linestyles):
            ax.plot(x, y, color = color, ls = linestyle)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.margins(x = 0.0, y = 0.1)

        # prepare the legends for the colors ...
        color_legend_elems = [Line2D([0], [0], marker = 's', color = 'none', markerfacecolor = cur_color, markeredgecolor = cur_color, label = cur_color_label) for cur_color, cur_color_label in color_labels.items()]
        color_legend_labels = color_labels.values()
        color_legend = ax.legend(handles = color_legend_elems, labels = color_legend_labels, loc = "upper left")

        # ... and for the line styles
        style_legend_elems = [Line2D([0], [0], marker = '', color = 'gray', markerfacecolor = 'gray', markeredgecolor = 'gray', label = cur_style_label, ls = cur_style) for cur_style, cur_style_label in linestyle_labels.items()]
        style_legend_labels = linestyle_labels.values()
        style_legend = ax.legend(handles = style_legend_elems, labels = style_legend_labels, loc = "upper right", ncol = 2)

        color_legend.get_frame().set_linewidth(0.0)
        style_legend.get_frame().set_linewidth(0.0)

        ax.add_artist(color_legend)
        ax.add_artist(style_legend)

        ax.set_ylim([0, 1.25 * ax.get_ylim()[1]])

        plt.tight_layout()
        fig.savefig(outfile)
        plt.close()

    @staticmethod
    def plot_significance_KS(sensdicts, outfile, 
                             model_SRs = ["significance_clf_tight_2J", "significance_clf_loose_2J", "significance_clf_tight_3J", "significance_clf_loose_3J"],
                             model_KSs = ["KS_bkg_class_tight_2J", "KS_bkg_class_loose_2J", "KS_bkg_class_tight_3J", "KS_bkg_class_loose_3J"],
                             reference_SRs = ["significance_low_MET_2J", "significance_high_MET_2J", "significance_low_MET_3J", "significance_high_MET_3J"],
                             reference_KSs = ["KS_bkg_low_MET_2J", "KS_bkg_high_MET_2J", "KS_bkg_low_MET_3J", "KS_bkg_high_MET_3J"]):

        # make sure to ge the color normalization correct
        colorquant = "lambda"
        cmap = plt.cm.viridis
        colorrange = [float(sensdict[colorquant]) for sensdict in sensdicts if colorquant in sensdict]
        norm = mpl.colors.Normalize(vmin = min(colorrange), vmax = max(colorrange))
        #norm = mpl.colors.Normalize(vmin = 0.0, vmax = 1.4)

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
            ax.plot(combined_sigs, KS_vals, color = color, linestyle = '-', marker = '_', alpha = 0.5)

        # also show the reference model
        sigs = [sensdict[cur_sig] for cur_sig in reference_SRs]
        combined_sig = np.sqrt(np.sum(np.square(sigs)))
        combined_sigs = np.full(len(sigs), combined_sig)

        # get the contributing KS values for the signal regions:
        KS_vals = [sensdict[cur_KS] for cur_KS in reference_KSs]
        ax.plot(combined_sigs, KS_vals, color = "firebrick", linestyle = '-', marker = '_', label = "cut-based analysis")

        # make colorbar for the range of encountered legended values
        cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.subplots_adjust(right = 0.8)
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap = cmap,
                                       norm = norm,
                                       orientation = 'vertical')
        cb.set_label(r'$\lambda$')

        ax.set_xlabel(r'expected sensitivity [$\sigma$]')
        ax.set_ylabel("KS (combined background)")
        ax.legend(loc = 'upper left')

        # use fixed plot ranges for the time being
        # ax.set_xlim([2.4, 2.7])
        # ax.set_ylim([0.0, 0.35])

        fig.savefig(outfile)
        plt.close()                

    @staticmethod
    def _perf_fairness_plot(perfdicts, xquant, KS_regex, colorquant, outpath, **kwargs):
        typ_perfdict = perfdicts[0]
        available_fields = typ_perfdict.keys()

        KS_fields = filter(KS_regex.match, available_fields)

        get_marker = lambda typestring: 'o'
        get_label = lambda typestring: typestring

        for KS_field in KS_fields:
            PerformancePlotter._perfdict_plot(perfdicts, xquant = xquant, yquant = KS_field, 
                                              colorquant = colorquant, markerquant = "adversary_model", markerstyle = get_marker,
                                              markerlabel = get_label, outfile = os.path.join(outpath, xquant + "_" + KS_field + ".pdf"), **kwargs)

    # worker method for flexible plotting: takes as inputs the list of perfdicts created by the ModelEvaluator
    # x/yquant ... quantity that should be printed along x/y
    # x/ylabel ... labels to be used for the x/y axis
    # colorquant ... quantity that controls the color
    # markerquant ... quantity that controls the type of marker that is to be used
    # markerstyle ... lambda of the form marker_style = lambda markerquant: return "marker_style"
    # markerlabel ... lambda of the form legend_entry = lambda markerquant: return "legend_entry"
    @staticmethod
    def _perfdict_plot(perfdicts, xquant, yquant, colorquant, markerquant, markerstyle, markerlabel, outfile, xlabel = "", ylabel = "", xlog = False, ylog = False, xaxis_range = [0.5, 1.0], yaxis_range = [0.0, 1.0], epilog = None, grid = True):
        cmap = plt.cm.viridis

        # find the proper normalization of the color map
        colorrange = [float(perfdict[colorquant]) for perfdict in perfdicts if colorquant in perfdict]
        norm = mpl.colors.Normalize(vmin = min(colorrange), vmax = max(colorrange))

        seen_types = []

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_axisbelow(True)
        ax.minorticks_on()
        
        if grid:
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

                #ax.scatter(perfdict[xquant], perfdict[yquant], color = color, label = label, marker = marker)
                ax.scatter(perfdict[xquant], perfdict[yquant], color = color, label = None, marker = marker, alpha = 0.5)

        # make colorbar for the range of encountered legended values
        cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.subplots_adjust(right = 0.8)
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap = cmap,
                                       norm = norm,
                                       orientation = 'vertical')
        cb.set_label(r'$\lambda$')

        # set limits
        ax.set_xlim(xaxis_range)
        ax.set_ylim(yaxis_range)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        if epilog:
            epilog(ax)

        # make legend with the different types that were encountered
        #legend_elems = [Line2D([0], [0], marker = markerstyle(cur_type), color = 'white', markerfacecolor = "black", label = markerlabel(cur_type)) for cur_type in seen_types]
        #ax.legend(handles = legend_elems)
        leg = ax.legend()
        leg.get_frame().set_linewidth(0.0)
        
        fig.savefig(outfile)
        plt.close()        

    @staticmethod
    def plot_AUROC_KS(perfdicts, outpath, colorquant = "lambda"):
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        PerformancePlotter._perf_fairness_plot(perfdicts, "AUROC", re.compile("KS_50_.*"), colorquant, outpath,
                                               xlabel = "AUROC", ylabel = "KS_50")
        PerformancePlotter._perf_fairness_plot(perfdicts, "AUROC", re.compile("KS_25_.*"), colorquant, outpath,
                                               xlabel = "AUROC", ylabel = "KS_50")

    @staticmethod
    def plot_background_rejection_JS(perfdicts, outpath, colorquant = "lambda"):
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # instruct it to draw shaded regions
        def shade_epilog_50(ax):
            ax.fill_between(x = [1, 100], y1 = [1, 1], y2 = [0.1, 0.1], facecolor = 'gray', alpha = 0.4)
            ax.fill_between(x = [0.1, 2], y1 = [1e4, 1e4], y2 = [1, 1], facecolor = 'gray', alpha = 0.4)
            ax.text(x = 3.0, y = 1.1, s = "maximal sculpting", rotation = 0, color = 'gray', size = 8)
            ax.text(x = 2.1, y = 15.0, s = "no separation", rotation = 90, color = 'gray', size = 8)

        def shade_epilog_25(ax):
            ax.fill_between(x = [1, 100], y1 = [1, 1], y2 = [0.1, 0.1], facecolor = 'gray', alpha = 0.4)
            ax.fill_between(x = [0.1, 4], y1 = [1e4, 1e4], y2 = [1, 1], facecolor = 'gray', alpha = 0.4)
            ax.text(x = 5.0, y = 1.1, s = "maximal sculpting", rotation = 0, color = 'gray', size = 8)
            ax.text(x = 4.1, y = 15.0, s = "no separation", rotation = 90, color = 'gray', size = 8)

        PerformancePlotter._perf_fairness_plot(perfdicts, "bkg_rejection_at_sigeff_50", re.compile("invJS_50_.*"), colorquant, outpath, 
                                               xlabel = r'$1/\epsilon_{\mathrm{bkg}}$ @ $\epsilon_{\mathrm{sig}} = 0.5$', ylabel = r'1/JSD @ $\epsilon_{\mathrm{sig}} = 0.5$',  
                                               xaxis_range = [1, 50], yaxis_range = [0.5, 1e4], xlog = True, ylog = True, epilog = shade_epilog_50, grid = False)
        PerformancePlotter._perf_fairness_plot(perfdicts, "bkg_rejection_at_sigeff_25", re.compile("invJS_25_.*"), colorquant, outpath, 
                                               xlabel = r'$1/\epsilon_{\mathrm{bkg}}$ @ $\epsilon_{\mathrm{sig}} = 0.25$', ylabel = r'1/JSD @ $\epsilon_{\mathrm{sig}} = 0.25$',
                                               xaxis_range = [1, 100], yaxis_range = [0.5, 1e4], xlog = True, ylog = True, epilog = shade_epilog_25, grid = False)   

    @staticmethod
    def plot_significance_fairness_inclusive(anadicts, outdir, colorquant = "lambda"):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        nJ = [2, 3]

        for cur_nJ in nJ:
            CBA_performance = anadicts[0]["{}jet_binned_sig_CBA".format(cur_nJ)]
            CBA_fairness = anadicts[0]["{}jet_high_low_MET_inv_JS_bkg".format(cur_nJ)]

            def CBA_epilog(ax):
                ax.text(x = 4, y = 2e3, s = "{} jet".format(cur_nJ))
                ax.scatter(CBA_performance, CBA_fairness, color = "firebrick", label = "cut-based analysis")
            
            xquant = "{}jet_binned_sig_PCA".format(cur_nJ)
            yquant = re.compile("{}jet_tight_loose_inv_JS_bkg".format(cur_nJ))
            PerformancePlotter._perf_fairness_plot(anadicts, xquant = xquant, KS_regex = yquant, xlabel = "binned significance", ylabel = "1/JSD",
                                                   colorquant = colorquant, outpath = outdir, yaxis_range = [0.5, 1e4], xaxis_range = [4, 6], ylog = True, epilog = CBA_epilog)

            xvals = [float(cur_dict["lambda"]) for cur_dict in anadicts]
            yvals = [float(cur_dict["{}jet_tight_loose_inv_JS_bkg".format(cur_nJ)]) for cur_dict in anadicts]
            yvals_2 = [float(cur_dict["loose_{}jet_inv_JS_bkg".format(cur_nJ)]) for cur_dict in anadicts]
            yvals_3 = [float(cur_dict["{}jet_binned_sig_PCA".format(cur_nJ)]) for cur_dict in anadicts]

            plt.scatter(xvals, yvals)
            plt.savefig(os.path.join(outdir, "lambda_traj_{}jet.pdf".format(cur_nJ)))
            plt.close()

            plt.scatter(xvals, yvals_2)
            plt.savefig(os.path.join(outdir, "lambda_traj_inclusive_{}jet.pdf".format(cur_nJ)))
            plt.close()

            plt.scatter(xvals, yvals_3)
            plt.savefig(os.path.join(outdir, "lambda_traj_comb_sig_{}jet.pdf".format(cur_nJ)))
            plt.close()
            
    @staticmethod
    def plot_significance_fairness_exclusive(anadicts, outdir, colorquant = "lambda"):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        nJ = [2, 3]
        cut_labels = ["tight", "loose"]
        CBA_labels = ["high_MET", "low_MET"]

        for cur_nJ, cur_axisrange in zip(nJ, [[2, 4.5], [0.5, 2.5]]):
            for cur_cut_label, cur_CBA_label in zip(cut_labels, CBA_labels):
                CBA_performance = anadicts[0]["{}_{}jet_binned_sig".format(cur_CBA_label, cur_nJ)]
                CBA_fairness = anadicts[0]["{}_{}jet_inv_JS_bkg".format(cur_CBA_label, cur_nJ)]

                def CBA_epilog(ax):
                    ax.text(x = 0.79, y = 0.85, s = "{}, {} jet".format(cur_cut_label, cur_nJ), transform = ax.transAxes)
                    ax.scatter(CBA_performance, CBA_fairness, color = "firebrick", label = "cut-based analysis")

                xquant = "{}_{}jet_binned_sig".format(cur_cut_label, cur_nJ)
                yquant = re.compile("{}_{}jet_inv_JS_bkg".format(cur_cut_label, cur_nJ))
                PerformancePlotter._perf_fairness_plot(anadicts, xquant = xquant, KS_regex = yquant, xlabel = r'binned significance [$\sigma$]', ylabel = "1/JSD",
                                                       colorquant = colorquant, outpath = outdir, yaxis_range = [0.5, 1e4], xaxis_range = cur_axisrange, ylog = True, epilog = CBA_epilog, grid = False)

    # all-in-one plotting of significances vs shaping in tight and loose SRs simultaneously, for each jet slice
    @staticmethod
    def plot_significance_fairness_combined(series_anadicts, series_cmaps, outdir, series_labels = [], nJ = 2, show_colorbar = False):
        colorquant = "lambda"

        fig = plt.figure(figsize = (10, 5))
        fig.subplots_adjust(right = 0.9, left = 0.08)
        ax = fig.add_subplot(111)

        if len(series_labels) != len(series_cmaps):
            series_labels = ["" for cur in series_cmaps]

        for anadicts, cmap in zip(series_anadicts, series_cmaps):

            # find the proper normalization of the color map
            colorrange = [float(anadict[colorquant]) for anadict in anadicts if colorquant in anadict]
            norm = mpl.colors.Normalize(vmin = min(colorrange), vmax = max(colorrange))

            for ind, anadict in enumerate(anadicts):
                color = cmap(norm(float(anadict[colorquant]))) if colorquant in anadict else "black"

                try:
                    ax.scatter(anadict["tight_{}jet_binned_sig".format(nJ)], anadict["tight_{}jet_inv_JS_bkg".format(nJ)], color = color, edgecolors = color, facecolors = color, linewidths = 1, label = None, marker = 'o', alpha = 1.0)
                    ax.scatter(anadict["loose_{}jet_binned_sig".format(nJ)], anadict["loose_{}jet_inv_JS_bkg".format(nJ)], color = color, edgecolors = color, facecolors = color, linewidths = 1, label = None, marker = '^', alpha = 1.0)

                    combined_sig = np.sqrt(anadict["loose_{}jet_binned_sig".format(nJ)] ** 2 + anadict["tight_{}jet_binned_sig".format(nJ)] ** 2)

                    # avoid falling out of the allowed range of the JSD, which can happen sometimes
                    # due to numerical problems for very strong shaping
                    if anadict["tight_{}jet_inv_JS_bkg".format(nJ)] < 1:
                        anadict["tight_{}jet_inv_JS_bkg".format(nJ)] = 1.0

                    ax.scatter(combined_sig, anadict["tight_{}jet_inv_JS_bkg".format(nJ)], color = color, facecolors = color, edgecolors = color, label = None, marker = 's', alpha = 1.0)
                except KeyError:
                    print(anadict)

            for prefix, label, mc, mfc, mec in zip(["original_", "optimized_"], ["", "(optimised)"], ["salmon", "white"], ["salmon", "white"], ["salmon", "salmon"]):
                ax.scatter(anadicts[0][prefix + "high_MET_{}jet_binned_sig".format(nJ)], 
                           anadicts[0][prefix + "high_MET_{}jet_inv_JS_bkg".format(nJ)], 
                           label = "cut-based analysis" + label, marker = 'o', color = mc, facecolors = mfc, edgecolors = mec)
            
                ax.scatter(anadicts[0][prefix + "low_MET_{}jet_binned_sig".format(nJ)],
                           anadicts[0][prefix + "low_MET_{}jet_inv_JS_bkg".format(nJ)], 
                           label = "cut-based analysis" + label, marker = '^', color = mc, facecolors = mfc, edgecolors = mec)
            
                combined_sig = np.sqrt(anadicts[0][prefix + "low_MET_{}jet_binned_sig".format(nJ)] ** 2 + anadicts[0][prefix + "high_MET_{}jet_binned_sig".format(nJ)] ** 2)
            
                ax.scatter(combined_sig,
                           anadicts[0][prefix + "high_MET_{}jet_inv_JS_bkg".format(nJ)], 
                           label = "cut-based analysis" + label, marker = 's', color = mc, facecolors = mfc, edgecolors = mec)

            if show_colorbar:
                cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                cb = mpl.colorbar.ColorbarBase(cb_ax, cmap = cmap,
                                               norm = norm,
                                               orientation = 'vertical')
                cb.set_label(r'$\lambda$')

        # prepare the legends for the CBA points
        legend_elems_CBA = [
            Line2D([0], [0], marker = '^', color = 'none', markerfacecolor = "salmon", markeredgecolor = "salmon", label = "low MET"),
            Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = "salmon", markeredgecolor = "salmon", label = "high MET"),
            Line2D([0], [0], marker = 's', color = 'none', markerfacecolor = "salmon", markeredgecolor = "salmon", label = "combined")
        ]
        legend_elems_CBA_optimized = [
            (Line2D([0], [0], marker = '^', color = 'none', markerfacecolor = "none", markeredgecolor = "salmon"),
             Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = "none", markeredgecolor = "salmon"),
             Line2D([0], [0], marker = 's', color = 'none', markerfacecolor = "none", markeredgecolor = "salmon", label = "optimized"))
        ]
        leg_labels_PCA = ["loose     ", "tight       ", "combined", r'$\lambda = 1.4$']
        leg_labels_CBA = [r'low-$E_{\mathrm{T}}^{\mathrm{miss}}$', r'high-$E_{\mathrm{T}}^{\mathrm{miss}}$', "combined"]

        leg_CBA = ax.legend(handles = legend_elems_CBA, labels = leg_labels_CBA, ncol = 3, framealpha = 0.0, columnspacing = 8.24, handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)}, loc = "upper left", bbox_to_anchor = (0.17, 0.20))
        leg_CBA_optimized = ax.legend(handles = legend_elems_CBA_optimized, labels = ["optimised"], ncol = 1, framealpha = 0.0, columnspacing = 0.1, handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)}, loc = "upper left", bbox_to_anchor = (0.17, 0.11))
        leg_CBA.get_frame().set_linewidth(0.0)
        leg_CBA_optimized.get_frame().set_linewidth(0.0)
        ax.add_artist(leg_CBA)
        ax.add_artist(leg_CBA_optimized)

        ax.text(x = 0.03, y = 0.07, s = "cut-based\n  analysis", transform = ax.transAxes)
        ax.text(x = 0.67, y = 0.88, s = r'$\sqrt{{s}}=13$ TeV, 140 fb$^{{-1}}$, {} jet'.format(nJ), transform = ax.transAxes)
        
        # prepare the legends for the individual PCA series
        for ind, (cmap, label) in enumerate(zip(series_cmaps, series_labels)):
            legend_elems_PCA = [
                Line2D([0], [0], marker = '^', color = 'none', markerfacecolor = cmap(200), markeredgecolor = cmap(200), label = "loose"),
                Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = cmap(200), markeredgecolor = cmap(200), label = "tight"),
                Line2D([0], [0], marker = 's', color = 'none', markerfacecolor = cmap(200), markeredgecolor = cmap(200), label = "combined"),
            ]        
            leg_PCA = ax.legend(handles = legend_elems_PCA, labels = leg_labels_PCA, ncol = 3, framealpha = 0.0, columnspacing = 8.5, handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)}, 
                                loc = "upper left", bbox_to_anchor = (0.17, 0.36 - 0.07 * ind))
            leg_PCA.get_frame().set_linewidth(0.0)
            ax.add_artist(leg_PCA)
            ax.text(x = 0.03, y = 0.27 - 0.07 * ind, s = label, transform = ax.transAxes)
            
        ax.set_xlim(left = ax.get_xlim()[0] * 0.7, right = ax.get_xlim()[1] * 1.05)
        ax.set_yscale("log")
        ax.set_xlabel(r'Binned significance [$\sigma$]')
        ax.set_ylabel(r'1/JSD')
        ax.set_ylim(bottom = 2e-2, top = 1e3)
        outfile = os.path.join(outdir, "{}jet_combined_JSD_sig.pdf".format(nJ))
        
        ax.axhline(y = 1.0, xmin = 0, xmax = 10, color = 'gray')
        ax.fill_between(x = ax.get_xlim(), y1 = [1, 1], y2 = [1e-3, 1e-3], facecolor = 'gray', alpha = 0.04)
        
        # now start putting the labels
        ax.text(x = 0.02, y = 0.62, s = r'less shaping $\rightarrow$', transform = ax.transAxes, rotation = 90, color = "gray")
        
        fig.savefig(outfile)
        plt.close()
                
    # @staticmethod
    # def _smooth_histogram(contents, edges, mode = "hist", npoints = 1000):
    #     from sklearn.neighbors import KernelDensity

    #     if mode == "hist":
    #         centres = (edges[:-1] + edges[1:]) / 2
    #     elif mode == "plot":
    #         centres = edges

    #     bw = (centres[1] - centres[0]) * 0.6

    #     kde = KernelDensity(bandwidth=bw)
    #     kde.fit(centres.reshape(-1, 1), sample_weight = contents)

    #     evalpts = np.linspace(centres[0], centres[-1], npoints)
    #     values = np.exp(kde.score_samples(evalpts))

    #     return evalpts, values

    @staticmethod
    def _smooth_histogram(contents, centres, npoints = 1000):
        import statsmodels.api as sm

        bw = (centres[1] - centres[0]) * 0.6

        dens = sm.nonparametric.KDEUnivariate(centres)
        dens.fit(fft = False, weights = contents.flatten(), bw = bw)

        evalpts = np.linspace(centres[0], centres[-1], npoints)
        values = dens.evaluate(evalpts)

        return evalpts, values

    # combine the passed plots and save them
    @staticmethod
    def combine_hists(perfdicts, hist_data, outpath, colorquant, plot_title, overlays = [], epilog = None, xlabel = "", ylabel = "", smoothing = False, cmap = plt.cm.Blues):

        # find the proper normalization of the color map
        if len(perfdicts) > 0:
            colorrange = [float(perfdict[colorquant]) for perfdict in perfdicts if colorquant in perfdict]
        else:
            colorrange = [0]
        norm = mpl.colors.Normalize(vmin = min(colorrange), vmax = max(colorrange))

        if len(perfdicts) != len(hist_data):
            raise Exception("need to get the same number of perfdicts and plots")

        bin_values = []
        bin_centers = []
        colors = []

        for perfdict, cur_hist in zip(perfdicts, hist_data):
            cur_bin_values = cur_hist[0]
            edges = cur_hist[1]
            xlabel_r = xlabel if xlabel else cur_hist[2]
            ylabel_r = ylabel if ylabel else cur_hist[3]

            color = cmap(norm(float(perfdict[colorquant]))) if colorquant in perfdict else "black"
            colors.append(color)

            low_edges = edges[:-1]
            high_edges = edges[1:]
            cur_centers = 0.5 * (low_edges + high_edges)

            if smoothing:
                cur_centers, cur_bin_values = PerformancePlotter._smooth_histogram(cur_bin_values, cur_centers)
            
            bin_centers.append(cur_centers)
            bin_values.append(cur_bin_values)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left = 0.15)

        # plot the combined histograms
        for cur_bin_centers, cur_bin_values, cur_color in zip(bin_centers, bin_values, colors):
            ax.plot(cur_bin_centers, cur_bin_values, color = cur_color, linewidth = 0.1)
        
        # plot the overlays
        for (x, y, opts) in overlays:
            if smoothing:
                x, y = PerformancePlotter._smooth_histogram(y, x)

            ax.plot(x, y, **opts)
            #leg = ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.05), ncol = 2)
            leg = ax.legend(loc = 'upper center', bbox_to_anchor = (0.74, 1.0), ncol = 1, framealpha = 0.0)
            for t in leg.texts:
                t.set_multialignment('left')
            leg.get_frame().set_linewidth(0.0)

        ax.set_xlabel(xlabel_r)
        ax.set_ylabel(ylabel_r)
        ax.margins(0.0)
        ax.set_title(plot_title)
        ax.set_ylim((0, 1.8 * ax.get_ylim()[1])) # add some more margin on top

        if epilog:
            epilog(ax)

        # make colorbar for the range of encountered legended values
        cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.subplots_adjust(right = 0.8)
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap = cmap,
                                       norm = norm,
                                       orientation = 'vertical')
        cb.set_label(r'$\lambda$')

        fig.savefig(outpath)
        plt.close()
