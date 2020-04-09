from argparse import ArgumentParser
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

class LossComparisonPlotter:

    def __init__(self, infile_A, infile_B):
        self.inpath_A = infile_A
        self.inpath_B = infile_B
        self.timeline_key = "batch"

        self.color_A = "black"
        self.color_B = "red"

    def compare(self, key_A, key_B, label_A, label_B, outfile, epilog = None):
        # load the available data
        with open(self.inpath_A, "rb") as infile_A, open(self.inpath_B, "rb") as infile_B:
            data_A = pickle.load(infile_A)
            data_B = pickle.load(infile_B)

        xcoord_A = data_A[self.timeline_key]
        xcoord_B = data_B[self.timeline_key]

        ycoord_A = data_A[key_A]
        ycoord_B = data_B[key_B]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(xcoord_A, ycoord_A, color = self.color_A, label = label_A)
        ax.plot(xcoord_B, ycoord_B, color = self.color_B, label = label_B)

        if epilog:
            epilog(ax)

        ax.legend()

        fig.savefig(outfile)

def MakePrivateDisCoLossComparison(DisCo_training_stats, training_stats, outfile):  

    # find the positions of the best-val-losses
    with open(DisCo_training_stats, "rb") as DisCo_file, open(training_stats, "rb") as other_file:
        data_DisCo = pickle.load(DisCo_file)
        data_other = pickle.load(other_file)

    best_DisCo_driven_DisCo_loss = data_DisCo["total_loss_validation"][np.argmin(data_DisCo["total_loss_validation"])]
    best_DisCo_driven_DisCo_loss_batch = data_DisCo["batch"][np.argmin(data_DisCo["total_loss_validation"])]
    best_other_driven_DisCo_loss = data_other["total_loss_private_DisCo_validation"][np.argmin(data_DisCo["total_loss_validation"])]
    best_other_driven_DisCo_loss_batch = data_other["batch"][np.argmin(data_DisCo["total_loss_validation"])]

    def best_val_loss_epilog(ax):
        ax.axhline(best_DisCo_driven_DisCo_loss, color = "black", ls = "--")
        ax.axhline(best_other_driven_DisCo_loss, color = "red", ls = "--")

        ax.plot(best_DisCo_driven_DisCo_loss_batch, best_DisCo_driven_DisCo_loss, 'ro', color = "black")
        ax.plot(best_other_driven_DisCo_loss_batch, best_other_driven_DisCo_loss, 'ro', color = "red")

    # load the training statistics from the DisCo-driven run and also the other one
    plotter = LossComparisonPlotter(infile_A = DisCo_training_stats, infile_B = training_stats)
    plotter.compare(key_A = "total_loss_validation", label_A = "DisCo-driven DisCo loss",
                    key_B = "total_loss_private_DisCo_validation", label_B = "MIND-driven DisCo loss",
                    outfile = outfile, epilog = best_val_loss_epilog)

if __name__ == "__main__":
    parser = ArgumentParser(description = "compare DisCo losses")
    parser.add_argument("--DisCo_training_stats", action = "store", dest = "DisCo_training_stats")
    parser.add_argument("--training_stats", action = "store", dest = "training_stats")
    parser.add_argument("--outfile", action = "store", dest = "outfile")
    args = vars(parser.parse_args())

    MakePrivateDisCoLossComparison(**args)
