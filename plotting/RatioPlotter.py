import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class RatioPlotter:

    # generates a ratio plot from two dumped histograms
    @staticmethod
    def histogram_ratio_plot(filepath_a, filepath_b, outfile, name_a = "", name_b = "", title = "", mode = 'np'):

        with open(filepath_a, 'rb') as file_a, open(filepath_b, 'rb') as file_b:
            # load the histogram contents from the files
            if mode == 'plt':
                (bin_contents_a, edges_a, _, _, _, _, _) = pickle.load(file_a)
                (bin_contents_b, edges_b, _, _, _, _, _) = pickle.load(file_b)
            elif mode == 'np':
                (bin_contents_a, edges_a, var_name_a) = pickle.load(file_a)
                (bin_contents_b, edges_b, var_name_b) = pickle.load(file_b)

            if not all(edges_a == edges_b):
                raise Exception("Error: require identical binnings for a ratio plot!")

            low_edges = edges_a[:-1]
            high_edges = edges_a[1:]
            centers = 0.5 * (low_edges + high_edges)

            fig = plt.figure()

            # first, plot the absolute values of the two histograms
            fig, ax = plt.subplots(2, 1, figsize = (5, 4), sharex = True, gridspec_kw = {'height_ratios': [2, 1]})
            ax[0].hist(centers, bins = edges_a, weights = bin_contents_a, label = name_a, histtype = 'step', color = "black", linestyle = "--")
            ax[0].hist(centers, bins = edges_a, weights = bin_contents_b, label = name_b, histtype = 'step', color = "red", linestyle = "-")
            ax[0].set_ylabel("normalized to 1")
            ax[0].tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
            ax[0].margins(0.0)
            ax[0].legend()

            ratio = np.nan_to_num(bin_contents_b / bin_contents_a)

            # then, also plot their ratio
            ax[1].hist(centers, bins = edges_a, weights = ratio, label = name_b, histtype = 'step', color = "red", linestyle = "-")
            ax[1].set_ylabel("MadGraph / ATLAS")
            ax[1].set_xlabel(var_name_a)
            ax[1].margins(0.0)
            ax[1].set_ylim([0.5, 1.5])
            ax[1].axhline(1.0, color = "black", linestyle = "--")
                
            fig.tight_layout()
            ax[0].set_title(title)
            plt.savefig(outfile)
            plt.close(fig)
