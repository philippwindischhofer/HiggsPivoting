import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class RatioPlotter:

    # generates a ratio plot from two dumped histograms
    @staticmethod
    def histogram_ratio_plot(filepath_a, filepath_b, outfile, name_a = "", name_b = "", mode = 'np'):

        with open(filepath_a, 'rb') as file_a, open(filepath_b, 'rb') as file_b:
            # load the histogram contents from the files
            if mode == 'plt':
                (bin_contents_a, edges_a, _, _, _, _) = pickle.load(file_a)
                (bin_contents_b, edges_b, _, _, _, _) = pickle.load(file_b)
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
            ax = fig.add_subplot(211)
            ax.hist(centers, weights = bin_contents_a, label = name_a, histtype = 'step')
            ax.hist(centers, weights = bin_contents_b, label = name_b, histtype = 'step')

            ratio = np.nan_to_num(bin_contents_b / bin_contents_a)

            print(ratio)

            # then, also plot their ratio
            ax = fig.add_subplot(212)
            ax.hist(centers, weights = ratio, label = name_b, histtype = 'step')
                
            plt.savefig(outfile)
            plt.close(fig)
