import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import os

class ModelEvaluator:

    def __init__(self, env):
        self.env = env

    # plot the ROC of the classifier
    def plot_roc(self, sig_data_test, bkg_data_test, outpath):
        pred_bkg = self.env.predict(data = bkg_data_test)[:,1]
        pred_sig = self.env.predict(data = sig_data_test)[:,1]

        pred = np.concatenate([pred_sig, pred_bkg], axis = 0)
        labels_test = np.concatenate([np.ones(len(pred_sig)), np.zeros(len(pred_bkg))], axis = 0)

        fpr, tpr, thresholds = metrics.roc_curve(labels_test, pred)
        auc = metrics.roc_auc_score(labels_test, pred)

        # plot the ROC
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, color = 'black')
        ax.set_xlabel("signal efficiency")
        ax.set_ylabel("background efficiency")
        ax.set_aspect(1.0)
        ax.text(0.75, 0.0, "AUC = {:.2f}".format(auc))
        plt.tight_layout()

        # create the output directory if it doesn't already exist and save the figure(s)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        fig.savefig(os.path.join(outpath, "ROC.pdf"))

    def plot_mBB_distortion(self, sig_data_test, bkg_data_test, outpath):
        pred_bkg = self.env.predict(data = bkg_data_test)[:,1]
        pred_sig = self.env.predict(data = sig_data_test)[:,1]

        # create the output directory if it doesn't already exist and save the figure(s)
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # plot the original mBB distributions for signal and background
        mBB_sig = sig_data_test["mBB"].values
        mBB_bkg = bkg_data_test["mBB"].values
        self._plot_mBB(mBB_sig, mBB_bkg, os.path.join(outpath, "distributions_raw.pdf"), label = [r'VHbb', r'$Z$ + jets'])

        # put some loose signal cut on the classifier
        mBB_sig_cut = mBB_sig[np.where(pred_sig > 0.5)]
        mBB_bkg_cut = mBB_bkg[np.where(pred_bkg > 0.5)]
        self._plot_mBB(mBB_sig_cut, mBB_bkg_cut, os.path.join(outpath, "distributions_cut_05.pdf"), label = [r'VHbb (class. > 0.5)', r'$Z$ + jets (class. > 0.5)'])

        # put some harsh signal cut on the classifier
        mBB_sig_cut = mBB_sig[np.where(pred_sig > 0.85)]
        mBB_bkg_cut = mBB_bkg[np.where(pred_bkg > 0.85)]
        self._plot_mBB(mBB_sig_cut, mBB_bkg_cut, os.path.join(outpath, "distributions_cut_085.pdf"), label = [r'VHbb (class. > 0.85)', r'$Z$ + jets (class. > 0.85)'])

    def _plot_mBB(self, mBB_sig, mBB_bkg, outfile, label):
        # plot the raw distributions
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist([mBB_sig, mBB_bkg], 300, density = True, histtype = 'step', stacked = False, fill = False, color = ["seagreen", "orange"], label = label)
        ax.legend()
        ax.set_xlim([0, 500])
        ax.set_xlabel(r'$m_{bb}$ [GeV]')
        ax.set_ylabel('a.u.')
        plt.tight_layout()
        fig.savefig(outfile) 

    # produce all performance plots
    def evaluate(self, sig_data_test, bkg_data_test, outpath):
        self.plot_roc(sig_data_test, bkg_data_test, outpath)
        self.plot_mBB_distortion(sig_data_test, bkg_data_test, outpath)
