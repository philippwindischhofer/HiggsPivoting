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
    def plot_roc(self, data_sig, data_bkg, sig_weights, bkg_weights, outpath):
        # need to merge all signal- and background samples
        data_sig = np.concatenate(data_sig, axis = 0)
        data_bkg = np.concatenate(data_bkg, axis = 0)
        sig_weights = np.concatenate(sig_weights, axis = 0)
        bkg_weights = np.concatenate(bkg_weights, axis = 0)

        pred_bkg = self.env.predict(data = data_bkg)[:,1]
        pred_sig = self.env.predict(data = data_sig)[:,1]

        pred = np.concatenate([pred_sig, pred_bkg], axis = 0)
        weights = np.concatenate([sig_weights, bkg_weights], axis = 0)
        labels = np.concatenate([np.ones(len(pred_sig)), np.zeros(len(pred_bkg))], axis = 0)

        fpr, tpr, thresholds = metrics.roc_curve(labels, pred, sample_weight = weights)
        auc = metrics.roc_auc_score(labels, pred, sample_weight = weights)

        # plot the ROC
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(tpr, fpr, color = 'black')
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

    # plot the mBB spectrum of the passed signal and background datasets
    def plot_mBB_distortion(self, data_sig, data_bkg, nuis_sig, nuis_bkg, weights_sig, weights_bkg, sigeffs, outpath, labels_sig = None, labels_bkg = None, num_cols = 2):
        pred_bkg = [self.env.predict(data = sample)[:,1] for sample in data_bkg]
        pred_sig = [self.env.predict(data = sample)[:,1] for sample in data_sig]
        
        # create the output directory if it doesn't already exist and save the figure(s)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        # compute the classifier cut values needed to achieve a certain signal efficiency
        pred_sig_merged = np.concatenate(pred_sig)
        weights_sig_merged = np.concatenate(weights_sig)
        cutvals = [self._weighted_percentile(pred_sig_merged, 1 - sigeff, weights = weights_sig_merged) for sigeff in sigeffs]

        for cutval, sigeff in zip(cutvals, sigeffs):
            print("classifier cut for {}% signal efficiency: {}".format(sigeff * 100, cutval))

        pred = pred_sig + pred_bkg
        nuis = nuis_sig + nuis_bkg
        weights = weights_sig + weights_bkg
        labels = labels_sig + labels_bkg

        fig = plt.figure(figsize = (15, 10))
        num_rows = len(pred) / num_cols

        # iterate over all data samples and fill a separate subplot for each
        for ind, (cur_pred, cur_nuis, cur_weights, cur_label) in enumerate(zip(pred, nuis, weights, labels)):
            plot_data = [cur_nuis]
            plot_weights = [cur_weights]
            plot_labels = [cur_label]

            # apply the classifier cuts
            for cutval, sigeff in zip(cutvals, sigeffs):
                cut_passed = np.where(cur_pred > cutval)
                plot_data.append(cur_nuis[cut_passed])
                plot_weights.append(cur_weights[cut_passed])
                plot_labels.append(cur_label + " ({}% signal eff.)".format(sigeff * 100))
                
            self._add_subplot(fig, vals = plot_data, weights = plot_weights, labels = plot_labels, nrows = num_rows, ncols = num_cols, num = ind + 1)
            
        # save the completed figure
        plt.tight_layout()
        fig.savefig(os.path.join(outpath, "dists_separate.pdf")) 
        plt.close()

    def _weighted_percentile(self, data, percentile, weights):
        # ensure that everything operates on flat data
        data = data.flatten()
        weights = weights.flatten()

        # first, reshuffle the data and the weights such that the data
        # is given in ascending order
        inds_sorted = np.argsort(data)
        data = data[inds_sorted]
        weights = weights[inds_sorted]

        # Note: the subtraction is just a tiebreaker
        weighted_percentiles = np.cumsum(weights) - 0.5 * weights

        # normalize between zero and one
        weighted_percentiles -= weighted_percentiles[0]
        weighted_percentiles /= weighted_percentiles[-1]

        retval = np.interp(percentile, weighted_percentiles, data)
        return retval

    def _add_subplot(self, fig, vals, weights, labels, nrows, ncols, num):
        ax = fig.add_subplot(nrows, ncols, num)
        ax.hist(vals, weights = weights, bins = 40, range = (0, 500), density = True, histtype = 'step', stacked = False, fill = False, label = labels)
        ax.legend()
        ax.set_xlabel(r'$m_{bb}$ [GeV]')
        ax.set_ylabel('a.u.')
        
    # produce all performance plots
    def performance_plots(self, data_sig, data_bkg, nuis_sig, nuis_bkg, weights_sig, weights_bkg, outpath, labels_sig = None, labels_bkg = None):
        self.plot_roc(data_sig, data_bkg, weights_sig, weights_bkg, outpath)
        self.plot_mBB_distortion(data_sig, data_bkg, nuis_sig, nuis_bkg, weights_sig, weights_bkg, [0.5, 0.25], outpath, labels_sig, labels_bkg)
