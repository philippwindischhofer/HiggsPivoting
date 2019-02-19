from base.Configs import TrainingConfig

import numpy as np

class Category:

    # need to store all the events in this category, depending on the process from which they came
    def __init__(self, name):
        self.name = name
        self.event_content = {}
        self.weight_content = {}

    def add_events(self, events, weights, process):
        if len(events) != len(weights):
            raise Exception("Need to have exactly one weight per event!")

        if not process in self.event_content:
            self.event_content[process] = events
            self.weight_content[process] = weights
        else:
            self.event_content[process] = np.append(self.event_content[process], events, axis = 0)
            self.weight_content[process] = np.append(self.weight_content[process], weights, axis = 0)

    # return the number of events coming from a certain process
    def get_number_events(self, process):
        if not process in self.weight_content:
            return 0.0

        return np.sum(self.weight_content[process])

    # compute the binned significance of the 'var' distribution of this category to the separation of the 
    # given signal- and background components
    def get_binned_significance(self, binning, signal_processes, background_processes, var_name):
        eps = 1e-5
        
        if not isinstance(binning, (list, np.ndarray)):
            raise Exception("Error: expect a list of explicit bin edges for this function!")

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

        invalid_mask = np.logical_or(total_binned_signal <= 0, total_binned_background <= 0)

        print("sig vs bkg")
        for bin_sig, bin_bkg in zip(total_binned_signal, total_binned_background):
            print("{} - {}".format(bin_sig, bin_bkg))
        
        # compute the binned significance
        binwise_significance = (total_binned_signal + total_binned_background) * np.log(1 + total_binned_signal / total_binned_background) - total_binned_signal
        binwise_significance[invalid_mask] = 0

        # exclude almost-empty bins
        # binwise_significance = np.nan_to_num(binwise_significance)
        # binwise_significance[binwise_significance == -np.inf] = 0
        # binwise_significance[binwise_significance == np.inf] = 0

        binned_sig = np.sqrt(2 * np.sum(binwise_significance, axis = 0))

        return binned_sig
