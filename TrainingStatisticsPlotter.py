import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle

class TrainingStatisticsPlotter:
    
    def __init__(self, indir):
        self.indir = indir
        self.timeline_key = "batch"

    def plot(self, outdir):
        # check if have the necessary training statistics file available in this directory
        try:
            with open(os.path.join(self.indir, "training_evolution.pkl"), "rb") as infile:
                stat_dict = pickle.load(infile)
                data_keys = [key for key in stat_dict.keys() if key != self.timeline_key]
                print("have the following timelines available for training statistics: " + ", ".join(data_keys))

                # generate plots for those timelines
                fig, axes = plt.subplots(nrows = len(data_keys), ncols = 1)
                fig.subplots_adjust(hspace = 0.5)

                for ind, (data_key, ax) in enumerate(zip(data_keys, axes)):
                    x_dat = stat_dict[self.timeline_key]
                    y_dat = stat_dict[data_key]
                    ax.plot(x_dat, y_dat, color = 'black')
                    ax.set_xlabel(self.timeline_key)
                    ax.set_ylabel(data_key)                    
                
                fig.savefig(os.path.join(outdir, "training_evolution.pdf"))
                
        except FileNotFoundError:
            print("no training statistics file found, skipping these plots...")
