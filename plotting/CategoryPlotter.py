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
    def plot_category_composition(category, outpath, process_order = ["Zjets", "Wjets", "singletop", "ttbar", "diboson", "Hbb"], var = "mBB", xlabel = "", ylabel = "events", plotlabel = [], args = {"range": (0, 500), "bins": 25}):
        colors = []
        data = []
        weights = []
        labels = []

        # choose some default ordering if no special choice is made
        if not process_order:
            process_order = category.event_content.keys()

        # go through the signal components that feed into this category and plot them as a stacked histogram
        for process_name in process_order:
            process_events = category.event_content[process_name]
            process_weights = category.weight_content[process_name]
            color = CategoryPlotter.process_colors[process_name]

            colors.append(color)
            data.append(process_events[:, TrainingConfig.training_branches.index(var)])
            weights.append(process_weights)
            labels.append(process_name)

        # then plot the histogram
        fig = plt.figure(figsize = (6, 5))
        ax = fig.add_subplot(111)
        n, bins, patches = ax.hist(data, weights = weights, histtype = 'step', fill = True, stacked = True, density = False, color = colors, label = labels, **args)
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # add the labels, if provided
        if plotlabel:
            text = "\n".join(plotlabel)
            plt.text(0.72, 0.95, text, verticalalignment = 'top', horizontalalignment = 'right', transform = ax.transAxes)

        plt.tight_layout()
        fig.savefig(outpath)
        plt.close()
        
        return n, bins, patches
