import os, re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
        #ax.set_ylim([1.3, 2.8])
        
        fig.savefig(outfile)
        plt.close()                

    @staticmethod
    def plot_asimov_significance_comparison(hypodicts, sensdicts, outdir, xlabel = r'$\lambda$', ylabel = r'expected significance [$\sigma_A$]',
                                        model_SRs = ["significance_clf_tight_2J", "significance_clf_loose_2J", "significance_clf_tight_3J", "significance_clf_loose_3J"]):

        assert len(hypodicts) == len(sensdicts) # make sure are given corresponding information
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

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

        PerformancePlotter._uncertainty_plot(lambdas, asimov_sigs_ncat_background_floating_mean, unc_up = asimov_sigs_ncat_background_floating_max - asimov_sigs_ncat_background_floating_mean, 
                                             unc_down = asimov_sigs_ncat_background_floating_min - asimov_sigs_ncat_background_floating_mean, 
                                             label = "pivotal classifier", outfile = os.path.join(outdir, "asimov_significance_background_floating.pdf"), xlabel = xlabel, ylabel = ylabel, color = 'royalblue', title = "background floating",
                                             epilog = lambda ax: ax.axhline(y = hypodict["asimov_sig_high_low_MET_background_floating"], xmin = 0.0, xmax = 1.0, color = 'royalblue', linestyle = "--", label = "cut-based analysis"))

        PerformancePlotter._uncertainty_plot(lambdas, asimov_sigs_ncat_background_fixed_mean, unc_up = asimov_sigs_ncat_background_fixed_max - asimov_sigs_ncat_background_fixed_mean, 
                                             unc_down = asimov_sigs_ncat_background_fixed_min - asimov_sigs_ncat_background_fixed_mean, 
                                             label = "pivotal classifier", outfile = os.path.join(outdir, "asimov_significance_background_fixed.pdf"), xlabel = xlabel, ylabel = ylabel, color = 'indianred', title = "background fixed",
                                             epilog = lambda ax: ax.axhline(y = hypodict["asimov_sig_high_low_MET_background_fixed"], xmin = 0.0, xmax = 1.0, color = 'indianred', linestyle = "--", label = "cut-based analysis"))

    @staticmethod
    def _uncertainty_plot(x, y, unc_up, unc_down, label, outfile, xlabel, ylabel, color, show_legend = True, epilog = None, title = ""):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # first, plot the central value
        ax.plot(x, y, marker = 'o', label = label, color = color)
        ax.fill_between(x, y + unc_down, y + unc_up, color = color, alpha = 0.4)

        if epilog is not None:
            epilog(ax)

        if show_legend:
            ax.legend(loc = 'best')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.margins(x = 0.0)
        
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
        ax.plot(combined_sigs, KS_vals, color = "tomato", linestyle = '-', marker = '_', label = "cut-based analysis")

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
                ax.scatter(CBA_performance, CBA_fairness, color = "tomato", label = "cut-based analysis")
            
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
                    ax.scatter(CBA_performance, CBA_fairness, color = "tomato", label = "cut-based analysis")

                xquant = "{}_{}jet_binned_sig".format(cur_cut_label, cur_nJ)
                yquant = re.compile("{}_{}jet_inv_JS_bkg".format(cur_cut_label, cur_nJ))
                PerformancePlotter._perf_fairness_plot(anadicts, xquant = xquant, KS_regex = yquant, xlabel = r'binned significance [$\sigma$]', ylabel = "1/JSD",
                                                       colorquant = colorquant, outpath = outdir, yaxis_range = [0.5, 1e4], xaxis_range = cur_axisrange, ylog = True, epilog = CBA_epilog, grid = False)

    # all-in-one plotting of significances vs shaping in tight and loose SRs simultaneously, for each jet slice
    @staticmethod
    def plot_significance_fairness_combined(anadicts, outdir, nJ = 2):
        cmap = plt.cm.viridis
        colorquant = "lambda"

        # find the proper normalization of the color map
        colorrange = [float(anadict[colorquant]) for anadict in anadicts if colorquant in anadict]
        norm = mpl.colors.Normalize(vmin = min(colorrange), vmax = max(colorrange))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for ind, anadict in enumerate(anadicts):
            color = cmap(norm(float(anadict[colorquant]))) if colorquant in anadict else "black"

            try:
                ax.scatter(anadict["tight_{}jet_binned_sig".format(nJ)], anadict["tight_{}jet_inv_JS_bkg".format(nJ)], color = color, label = None, marker = 'o', alpha = 0.5)
                ax.scatter(anadict["loose_{}jet_binned_sig".format(nJ)], anadict["loose_{}jet_inv_JS_bkg".format(nJ)], color = color, label = None, marker = '+', alpha = 0.5)
            except KeyError:
                print(anadict)

        ax.scatter(anadicts[0]["high_MET_{}jet_binned_sig".format(nJ)], 
                   anadicts[0]["high_MET_{}jet_inv_JS_bkg".format(nJ)], 
                   color = "tomato", label = "cut-based analysis", marker = 'o')

        ax.scatter(anadicts[0]["low_MET_{}jet_binned_sig".format(nJ)],
                   anadicts[0]["low_MET_{}jet_inv_JS_bkg".format(nJ)], 
                   color = "tomato", label = "cut-based analysis", marker = '+')

        ax.set_yscale("log")
        ax.set_xlabel(r'binned signficance [$\sigma$]')
        ax.set_ylabel(r'1/JSD')
        ax.set_ylim(bottom = 0.5)
        outfile = os.path.join(outdir, "{}jet_combined_JSD_sig.pdf".format(nJ))
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
    def combine_hists(perfdicts, hist_data, outpath, colorquant, plot_title, overlays = [], epilog = None, xlabel = "", ylabel = "", smoothing = False):
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

        # plot the combined histograms
        for cur_bin_centers, cur_bin_values, cur_color in zip(bin_centers, bin_values, colors):
            ax.plot(cur_bin_centers, cur_bin_values, color = cur_color, linewidth = 0.1)

        ax.set_xlabel(xlabel_r)
        ax.set_ylabel(ylabel_r)
        ax.margins(0.0)
        ax.set_title(plot_title)
        ax.set_ylim((0, 1.2 * ax.get_ylim()[1])) # add some more margin on top

        if epilog:
            epilog(ax)
        
        # plot the overlays
        for (x, y, opts) in overlays:
            if smoothing:
                x, y = PerformancePlotter._smooth_histogram(y, x)

            ax.plot(x, y, **opts)
            leg = ax.legend()
            leg.get_frame().set_linewidth(0.0)

        # make colorbar for the range of encountered legended values
        cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.subplots_adjust(right = 0.8)
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap = cmap,
                                       norm = norm,
                                       orientation = 'vertical')
        cb.set_label(r'$\lambda$')

        fig.savefig(outpath)
        plt.close()
