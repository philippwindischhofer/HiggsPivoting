import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from base.Configs import TrainingConfig

class CategoryPlotter:
    
    # try to be compatible with the official colors
    process_colors = {"Hbb": "#ff0000ff", 
                      "ttbar": "#ffcc00ff", 
                      "singletop": "#cc9700ff", 
                      "Wjets": "#006300ff", 
                      "Zjets": "#0063ccff", 
                      "diboson": "#ccccccff"}

    # use the events in the given category to plot the spectrum of a certain event variable
    @staticmethod
    def plot_category_composition(category, binning, outpath, process_order = ["Zjets", "Wjets", "singletop", "ttbar", "diboson", "Hbb"], var = "mBB", xlabel = "", ylabel = "events", plotlabel = [], args = {}, logscale = False, ignore_binning = False):
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
            labels.append(process_name)

        # then plot the histogram
        fig = plt.figure(figsize = (6, 5))
        ax = fig.add_subplot(111)
        n, bins, patches = ax.hist(data, weights = weights, histtype = 'stepfilled', stacked = True, color = colors, label = labels, bins = binning, **args)
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.margins(0.0)

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
