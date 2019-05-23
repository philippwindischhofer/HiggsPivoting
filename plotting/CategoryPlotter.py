import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from base.Configs import TrainingConfig

class CategoryPlotter:
    
    # try to be compatible with the official colors
    process_colors = {"Hbb": "#ff0000", 
                      "ttbar": "#ffcc00", 
                      "singletop": "#cc9700", 
                      "Wjets": "#006300", 
                      "Zjets": "#0063cc", 
                      "diboson": "#cccccc",
                      "generic_process": "#726c6a"}

    process_labels = {"Hbb": r'$VH$, $H\rightarrow b\bar{b}$',
                      "ttbar": r'$t\bar{t}$',
                      "Wjets": r'$W$ + jets',
                      "Zjets": r'$Z$ + jets',
                      "diboson": r'Diboson'
    }

    # use the events in the given category to plot the spectrum of a certain event variable
    @staticmethod
    def plot_category_composition(category, binning, outpath, process_order = TrainingConfig.bkg_samples + TrainingConfig.sig_samples, var = "mBB", xlabel = "", ylabel = "events", plotlabel = [], args = {}, logscale = False, ignore_binning = False, histtype = 'stepfilled', stacked = True, density = False):
        if not isinstance(binning, (list, np.ndarray)):
            raise Exception("Error: expect a list of explicit bin edges for this function!")
        
        colors = []
        data = []
        weights = []
        labels = []

        # choose some default ordering if no special choice is made
        if not process_order:
            process_order = category.event_content.keys()

        # go through the signal components that feed into this category and plot them as a stacked histogram
        for process_name in process_order:
            process_values, process_weights = category.get_event_variable(process_name, var)

            color = CategoryPlotter.process_colors[process_name]

            colors.append(color)
            clipped_values = np.clip(process_values, binning[0], binning[-1])
            data.append(clipped_values)
            weights.append(process_weights)
            labels.append(CategoryPlotter.process_labels[process_name])

        # then plot the histogram
        fig = plt.figure(figsize = (6, 5))
        ax = fig.add_subplot(111)

        centers = []
        bin_contents = []
        sow_squared_total = np.zeros(len(binning) - 1)

        for cur_data, cur_weights in zip(data, weights):

            cur_bin_contents, cur_bin_edges = np.histogram(cur_data, bins = binning, weights = cur_weights.flatten(), density = density)
            bins = np.digitize(cur_data, bins = binning) - 1 # subtract 1 to get back a 0-indexed array

            # compute sum-of-weights-squared for each bin to get the uncertainties correct
            sow_squared = np.array([np.sum(np.square(cur_weights[np.argwhere(bins == cur_bin)])) for cur_bin in range(0, len(binning) - 1)]).flatten()

            sow_squared_total = np.add(sow_squared_total, sow_squared)

            if ignore_binning:
                cur_bin_edges = np.linspace(cur_bin_edges[0], cur_bin_edges[-1], num = len(cur_bin_edges), endpoint = True)

            lower_edges = cur_bin_edges[:-1]
            upper_edges = cur_bin_edges[1:]

            cur_centers = np.array(0.5 * (lower_edges + upper_edges))

            centers.append(cur_centers)
            bin_contents.append(cur_bin_contents)

        # compute the actual per-bin uncertainty
        sow_total = np.sqrt(sow_squared_total)
        
        #n, bins, patches = ax.hist(data, weights = weights, histtype = 'stepfilled', stacked = True, color = colors, label = labels, bins = binning, **args)
        n, bins, patches = ax.hist(centers, weights = bin_contents, histtype = histtype, stacked = stacked, color = colors, edgecolor = 'black', label = labels, bins = cur_bin_edges, **args)

        if stacked:
            error_centers = centers[0]
            error_offset = np.sum(bin_contents, axis = 0)
            ax.errorbar(error_centers, error_offset, yerr = sow_total, fmt = 'k', linestyle = 'None')

        leg = ax.legend(loc = "upper right")
        leg.get_frame().set_linewidth(0.0)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.margins(0.0)

        # add some margin on top of the figures
        ax.set_ylim([0, 1.3 * ax.get_ylim()[1]])

        if logscale:
            plt.yscale('log')

        # add the labels, if provided
        if plotlabel:
            text = "\n".join(plotlabel)
            plt.text(0.72, 0.95, text, verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

        plt.tight_layout()
        fig.savefig(outpath)
        plt.close()
        
        return n, bins, patches
