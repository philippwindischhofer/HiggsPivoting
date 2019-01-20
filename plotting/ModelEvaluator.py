import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import mutual_info_regression

class ModelEvaluator:

    def __init__(self, env):
        self.env = env

    # computes some characteristics of the data directly, no model involved
    @staticmethod
    def get_data_metrics(sig_data_test, bkg_data_test):
        # now add the purely data-based information as well
        datadict = {"lambdaleglabel": "data"}

        mBB_sig = sig_data_test["mBB"].values
        mBB_bkg = bkg_data_test["mBB"].values
        mBB = np.concatenate([mBB_sig, mBB_bkg])
        
        label_sig = np.ones(len(mBB_sig))
        label_bkg = np.zeros(len(mBB_bkg))
        labels = np.concatenate([label_sig, label_bkg])

        data = np.concatenate([sig_data_test.values, bkg_data_test.values], axis = 0)

        datadict["logI(f,label)"] = np.log(mutual_info_regression(data, labels.ravel(), copy = False))
        datadict["logI(f,nu)"] = np.log(mutual_info_regression(data, mBB.ravel(), copy = False))

        # datadict["logI(f,label)"] = np.random.rand()
        # datadict["logI(f,nu)"] = np.random.rand()
        
        datadict["type"] = "data"

        return datadict

    # computes a series of performance measures: ROC AUC
    def get_performance_metrics(self, sig_data_test, bkg_data_test):
        retdict = {}

        mBB_sig = sig_data_test["mBB"].values
        mBB_bkg = bkg_data_test["mBB"].values
        mBB = np.concatenate([mBB_sig, mBB_bkg])

        # get the model's predictions
        pred_bkg = self.env.predict(data = bkg_data_test)[:,1]
        pred_sig = self.env.predict(data = sig_data_test)[:,1]
        pred = np.concatenate([pred_sig, pred_bkg], axis = 0)
        pred = np.expand_dims(pred, axis = 1)

        labels_test = np.concatenate([np.ones(len(pred_sig)), np.zeros(len(pred_bkg))], axis = 0)

        # get ROC AUC
        retdict["ROCAUC"] = metrics.roc_auc_score(labels_test, pred)

        # get low-level measure for how well the mBB distributions are preserved
        mBB_sig_cut = mBB_sig[np.where(pred_sig > 0.85)]
        mBB_bkg_cut = mBB_bkg[np.where(pred_bkg > 0.85)]
        mBB_sig_cut_binned, _ = np.histogram(mBB_sig_cut, bins = 100, range = (0, 500), density = True)
        mBB_bkg_cut_binned, _ = np.histogram(mBB_bkg_cut, bins = 100, range = (0, 500), density = True)

        mBB_sig_binned, _ = np.histogram(mBB_sig, bins = 100, range = (0, 500), density = True)
        mBB_bkg_binned, _ = np.histogram(mBB_bkg, bins = 100, range = (0, 500), density = True)

        # compute the squared differences between the raw, original distribution and the ones after the cut has been applied
        mBB_sig_sq_diff = np.sqrt(np.sum(np.square(mBB_sig_cut_binned - mBB_sig_binned)))
        mBB_bkg_sq_diff = np.sqrt(np.sum(np.square(mBB_bkg_cut_binned - mBB_bkg_binned)))
        
        retdict["sig_sq_diff"] = mBB_sig_sq_diff
        retdict["bkg_sq_diff"] = mBB_bkg_sq_diff

        # get mutual information between prediction and true class label
        #retdict["logI(f,label)"] = np.log(mutual_info_regression(pred, labels_test, copy = False)[0])
        retdict["logI(f,label)"] = np.random.rand()

        # get mutual information between prediction and nuisance
        #retdict["logI(f,nu)"] = np.log(mutual_info_regression(pred, mBB, copy = False)[0])
        retdict["logI(f,nu)"] = np.random.rand()

        # get additional information about this model and add it - may be important for plotting later
        for key, val in self.env.global_pars.items():
            retdict[key] = val

        # possibly prepare labels for important sweep quantities
        if "lambda" in retdict:
            retdict["lambdaleglabel"] = r'$\lambda = {:.2f}$'.format(float(retdict["lambda"]))

        return retdict

    # plot the ROC of the classifier
    def plot_roc(self, sig_data_test, bkg_data_test, outpath):
        # need to merge all signal- and background samples
        sig_data_test = pd.concat(sig_data_test)
        bkg_data_test = pd.concat(bkg_data_test)

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
        plt.close()

    def plot_mBB_distortion(self, sig_data_test, bkg_data_test, outpath, sig_labels = None, bkg_labels = None):
        pred_bkg = [self.env.predict(data = sample)[:,1] for sample in bkg_data_test]
        pred_sig = [self.env.predict(data = sample)[:,1] for sample in sig_data_test]

        # create the output directory if it doesn't already exist and save the figure(s)
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # plot the original mBB distributions for signal and background
        mBB_sig = [sample["mBB"].values for sample in sig_data_test]
        mBB_bkg = [sample["mBB"].values for sample in bkg_data_test]

        # find the cut values that lead to a certain signal efficiency
        cutval_sigeff_50 = np.percentile(pred_sig, 50)
        cutval_sigeff_25 = np.percentile(pred_sig, 75)

        print("classifier cut for 50% signal efficiency: {}".format(cutval_sigeff_50))
        print("classifier cut for 25% signal efficiency: {}".format(cutval_sigeff_25))

        # put some loose signal cut on the classifier
        mBB_sig_cut_50 = [mBB[np.where(pred > cutval_sigeff_50)] for mBB, pred in zip(mBB_sig, pred_sig)]
        mBB_bkg_cut_50 = [mBB[np.where(pred > cutval_sigeff_50)] for mBB, pred in zip(mBB_bkg, pred_bkg)]

        # put some harsh signal cut on the classifier
        mBB_sig_cut_25 = [mBB[np.where(pred > cutval_sigeff_25)] for mBB, pred in zip(mBB_sig, pred_sig)]
        mBB_bkg_cut_25 = [mBB[np.where(pred > cutval_sigeff_25)] for mBB, pred in zip(mBB_bkg, pred_bkg)]

        to_plot = [[data for data in WPs] for WPs in zip(mBB_sig + mBB_bkg,
                                                         mBB_sig_cut_50 + mBB_bkg_cut_50,
                                                         mBB_sig_cut_25 + mBB_bkg_cut_25)]
        labels = [[label for label in WP_label] for WP_label in zip(sig_labels + bkg_labels,
                                                                    [sample_name + " (50% signal eff.)" for sample_name in sig_labels + bkg_labels],
                                                                    [sample_name + " (25% signal eff.)" for sample_name in sig_labels + bkg_labels])]

        self._plot_mBB(to_plot, 
                       os.path.join(outpath, "distributions_combined.pdf"), 
                       labels = labels)
        
    def _plot_mBB(self, vals, outfile, labels, num_cols = 2):
        num_rows = len(vals) / num_cols

        # plot the raw distributions
        fig = plt.figure(figsize = (15, 10))

        for ind, (cur_val, cur_label) in enumerate(zip(vals, labels)):
            ax = fig.add_subplot(num_rows, num_cols, ind + 1)
            ax.hist(cur_val, 300, density = True, histtype = 'step', stacked = False, fill = False, label = cur_label)
            ax.legend()
            ax.set_xlim([0, 500])
            ax.set_xlabel(r'$m_{bb}$ [GeV]')
            ax.set_ylabel('a.u.')

        plt.tight_layout()
        fig.savefig(outfile) 
        plt.close()

    # produce all performance plots
    def performance_plots(self, sig_data_test, bkg_data_test, outpath, sig_labels = None, bkg_labels = None):
        self.plot_roc(sig_data_test, bkg_data_test, outpath)
        self.plot_mBB_distortion(sig_data_test, bkg_data_test, outpath, sig_labels, bkg_labels)

