import numpy as np
import pandas as pd
from analysis.Category import Category
from base.Configs import TrainingConfig
from plotting.ModelEvaluator import ModelEvaluator
from training.DataFormatters import TrainingSample

class ClassifierBasedCategoryFiller:

    @staticmethod
    def _sigeff_range_to_score_range(all_signal_pred, all_signal_weights, sigeff_range):
        return (ModelEvaluator._weighted_percentile(all_signal_pred, 1 - sigeff_range[0], weights = all_signal_weights), 
                ModelEvaluator._weighted_percentile(all_signal_pred, 1 - sigeff_range[1], weights = all_signal_weights))
        
    @staticmethod
    def create_classifier_category(mcoll, sig_process_data, sig_process_names, bkg_process_data, bkg_process_names, classifier_sigeff_range = (1.0, 0.0), nJ = 2):
        
        # make sure to base all selections only on signal events with the correct number of jets
        sig_process_data = [cur_data.loc[cur_data["nJ"] == nJ] for cur_data in sig_process_data]
        bkg_process_data = [cur_data.loc[cur_data["nJ"] == nJ] for cur_data in bkg_process_data]
        
        # convert them to TrainingSamples as well
        sig_process_TrainingSamples = [TrainingSample.fromTable(cur_data) for cur_data in sig_process_data]
        bkg_process_TrainingSamples = [TrainingSample.fromTable(cur_data) for cur_data in bkg_process_data]
        all_signal_TrainingSample = TrainingSample.fromTable(pd.concat(sig_process_data))

        # obtain the classifier predictions on all samples
        sig_process_preds = [mcoll.predict(cur_data)[:, 1] for cur_data in sig_process_data]
        bkg_process_preds = [mcoll.predict(cur_data)[:, 1] for cur_data in bkg_process_data]
        all_signal_pred = np.concatenate(sig_process_preds, axis = 0)

        # first, determine the cuts on the classifier based on the asked-for signal efficiency
        classifier_range = ClassifierBasedCategoryFiller._sigeff_range_to_score_range(all_signal_pred, all_signal_weights = all_signal_TrainingSample.weights, sigeff_range = classifier_sigeff_range)
        print("translated signal efficiency range ({}, {}) to classifier output range ({}, {})".format(classifier_sigeff_range[0], classifier_sigeff_range[1], 
                                                                                                       classifier_range[0], classifier_range[1]))
        
        retcat = Category("clf_{:.2f}_{:.2f}".format(classifier_sigeff_range[0], classifier_sigeff_range[1]))

        # then fill all events from all signal + background processes
        process_data = sig_process_data + bkg_process_data
        process_names = sig_process_names + bkg_process_names
        process_preds = sig_process_preds + bkg_process_preds

        for cur_process_data, cur_process_name, cur_pred in zip(process_data, process_names, process_preds):
            
            print("predicting on sample {} with length {}".format(cur_process_name, len(cur_process_data)))

            cut = np.logical_and.reduce((cur_pred > classifier_range[0], cur_pred < classifier_range[1]))

            assert len(cut) == len(cur_process_data)
            passed = cur_process_data[cut]
            passed = TrainingSample.fromTable(passed)
            
            # fill the category
            retcat.add_events(events = passed.data, weights = passed.weights, process = cur_process_name, event_variables = TrainingConfig.training_branches)

            print("filled {} events from process '{}'".format(sum(passed.weights), cur_process_name))

        return retcat
