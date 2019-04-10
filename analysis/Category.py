from base.Configs import TrainingConfig

import pickle
import numpy as np
from array import array

# for exporting ROOT histograms
import ROOT
from ROOT import TH1F, TFile

class Category:

    # need to store all the events in this category, depending on the process from which they came
    def __init__(self, name):
        self.name = name
        self.event_content = {}
        self.weight_content = {}
        self.event_variables = {}

        # holds some auxiliary information that is not part of the event per se
        self.aux_content = {}
        self.aux_variables = {}

    def add_events(self, events, weights, process, event_variables, aux_content = None, aux_variables = None):
        if len(events) != len(weights):
            raise Exception("Need to have exactly one weight per event!")

        if not process in self.event_content:
            self.event_content[process] = events
            self.weight_content[process] = weights
            self.event_variables[process] = event_variables
        else:
            self.event_content[process] = np.append(self.event_content[process], events, axis = 0)
            self.weight_content[process] = np.append(self.weight_content[process], weights, axis = 0)

        if not process in self.aux_content:
            self.aux_content[process] = aux_content
            self.aux_variables[process] = aux_variables
        else:
            self.aux_content[process] = np.append(self.aux_content[process], aux_content, axis = 0)
            
    # return the number of events coming from a certain process
    def get_number_events(self, process):
        if not process in self.weight_content:
            return 0.0

        return np.sum(self.weight_content[process])

    def get_event_variable(self, processes, var):
        if not isinstance(processes, list):
            processes = [processes]

        event_retval = []
        weight_retval = []

        for process in processes:
            # check if this variable is part of the event- or the auxiliary information:
            event_vars = self.event_variables[process]
            aux_vars = self.aux_variables[process]

            if var in event_vars:
                event_retval.append(self.event_content[process][:, event_vars.index(var)])
            elif var in aux_vars:
                event_retval.append(self.aux_content[process][:, aux_vars.index(var)])
            else:
                raise KeyError("Error: unknown variable '{}'".format(var))                                    

            weight_retval.append(self.weight_content[process])

        event_retval = np.concatenate(event_retval)
        weight_retval = np.concatenate(weight_retval)

        return event_retval, weight_retval

    # computes and exports the histogram of some event variable, as filled in this category
    def export_histogram(self, binning, processes, var_name, outfile, clipping = False, density = True):
        # obtain the data that is to be plotted
        data, weights = self.get_event_variable(processes, var_name)

        # perform the histogramming
        if clipping:
            data = np.clip(data, binning[0], binning[-1])
        
        n, bins = np.histogram(data, bins = binning, weights = weights.flatten(), density = density)

        with open(outfile, "wb") as outfile:
            pickle.dump((n, bins, var_name), outfile)

    # similar to 'export_histogram', but instead writes a *.root file
    def export_ROOT_histogram(self, binning, processes, var_names, outfile_path, clipping = False, density = False, ignore_binning = False):
        if isinstance(var_names, list):
            if len(var_names) > 1:
                raise NotImplementedError("Error: can only export TH1 up to now - please call for a single variable at a time!")
            else:
                var_name = var_names[0]
        else:
            var_name = var_names # just to get the semantics right :)

        outfile = TFile(outfile_path, 'RECREATE')
        ROOT.SetOwnership(outfile, False)
        outfile.cd()

        for process in processes:
            # obtain the data that is to be histogrammed
            data, weights = self.get_event_variable(process, var_name)

            # perform the histogramming
            if clipping:
                data = np.clip(data, binning[0], binning[-1])

            bin_contents, bins = np.histogram(data, bins = binning, weights = weights.flatten(), density = density)

            # now, just need to fill it into a ROOT histogram and dump it into a file
            hist_name = process + "_" + var_name
            if ignore_binning:
                hist = TH1F(hist_name, hist_name, len(bins) - 1, bins[0], bins[-1])
            else:
                hist = TH1F(hist_name, hist_name, len(bins) - 1, array('d', bins))
            ROOT.SetOwnership(hist, False) # avoid problems with Python's garbage collector

            for bin_number, bin_content in enumerate(bin_contents):
                if bin_content <= 0:
                    bin_content = 1e-4

                hist.SetBinContent(bin_number + 1, bin_content)

            hist.Write()

        outfile.Close()

    def _get_SB_binning(self, binning, signal_processes, background_processes, var_name):
        # first, bin all participating processes
        binned_signal = []
        binned_background = []

        for process_name, events in self.event_content.items():
            cur_var_values = events[:, TrainingConfig.training_branches.index(var_name)]
            cur_weights = self.weight_content[process_name]

            cur_bin_contents, _ = np.histogram(np.clip(cur_var_values, binning[0], binning[-1]), bins = binning, weights = cur_weights.flatten())

            if process_name in signal_processes:
                binned_signal.append(cur_bin_contents)
            elif process_name in background_processes:
                binned_background.append(cur_bin_contents)

        # get the total sum of signal- and background events in each bin
        total_binned_signal = np.sum(binned_signal, axis = 0)
        total_binned_background = np.sum(binned_background, axis = 0)

        return total_binned_signal, total_binned_background

    # compute the binned significance of the 'var' distribution of this category to the separation of the 
    # given signal- and background components
    def get_binned_significance(self, binning, signal_processes, background_processes, var_name, verbose = False):
        eps = 1e-5
        
        if not isinstance(binning, (list, np.ndarray)):
            raise Exception("Error: expect a list of explicit bin edges for this function!")

        total_binned_signal, total_binned_background = self._get_SB_binning(binning, signal_processes, background_processes, var_name)

        # exclude almost-empty bins
        invalid_mask = np.logical_or(total_binned_signal <= 0, total_binned_background <= 0)

        if verbose:
            print("sig vs bkg")
            for bin_sig, bin_bkg in zip(total_binned_signal, total_binned_background):
                print("{} - {}".format(bin_sig, bin_bkg))
        
        # compute the binned significance
        binwise_significance = (total_binned_signal + total_binned_background) * np.log(1 + total_binned_signal / total_binned_background) - total_binned_signal
        binwise_significance[invalid_mask] = 0

        binned_sig = np.sqrt(2 * np.sum(binwise_significance, axis = 0))

        return binned_sig

    # return s / sqrt(s + b) summed in quadrature for all bins
    def get_S_sqrt_SB(self, binning, signal_processes, background_processes, var_name):
        eps = 1e-5
        
        if not isinstance(binning, (list, np.ndarray)):
            raise Exception("Error: expect a list of explicit bin edges for this function!")

        total_binned_signal, total_binned_background = self._get_SB_binning(binning, signal_processes, background_processes, var_name)

        # exclude almost-empty bins
        invalid_mask = np.logical_or(total_binned_signal <= 0, total_binned_background <= 0)

        print("sig vs bkg")
        for bin_sig, bin_bkg in zip(total_binned_signal, total_binned_background):
            print("{} - {}".format(bin_sig, bin_bkg))
        
        # compute the binned significance
        S_sqrt_SB = total_binned_signal / np.sqrt(total_binned_signal + total_binned_background)
        S_sqrt_SB[invalid_mask] = 0

        retval = np.sqrt(np.sum(np.square(S_sqrt_SB), axis = 0))

        return retval

    # return s / sqrt(b) summed in quadrature for all bins
    def get_S_sqrt_B(self, binning, signal_processes, background_processes, var_name):
        eps = 1e-5
        
        if not isinstance(binning, (list, np.ndarray)):
            raise Exception("Error: expect a list of explicit bin edges for this function!")

        total_binned_signal, total_binned_background = self._get_SB_binning(binning, signal_processes, background_processes, var_name)

        # exclude almost-empty bins
        invalid_mask = np.logical_or(total_binned_signal <= 0, total_binned_background <= 0)

        print("sig vs bkg")
        for bin_sig, bin_bkg in zip(total_binned_signal, total_binned_background):
            print("{} - {}".format(bin_sig, bin_bkg))
        
        # compute the binned significance
        S_sqrt_B = total_binned_signal / np.sqrt(total_binned_background)
        S_sqrt_B[invalid_mask] = 0

        retval = np.sqrt(np.sum(np.square(S_sqrt_B), axis = 0))

        return retval
        
