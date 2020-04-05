import numpy as np
import pandas as pd
from analysis.Category import Category
from base.Configs import TrainingConfig
from plotting.ModelEvaluator import ModelEvaluator
from training.DataFormatters import TrainingSample

class ClassifierBasedCategoryFiller:

    @staticmethod
    def _sigeff_range_to_score_range(mcoll, sig_process_data, sigeff_range):
        all_signal_data = pd.concat(sig_process_data)
        all_signal_weights = TrainingSample.fromTable(all_signal_data).weights
        all_signal_pred = mcoll.predict(all_signal_data)[:, 1]
        return (ModelEvaluator._weighted_percentile(all_signal_pred, 1 - sigeff_range[0], weights = all_signal_weights), 
                ModelEvaluator._weighted_percentile(all_signal_pred, 1 - sigeff_range[1], weights = all_signal_weights))
        
    @staticmethod
    def create_classifier_category(mcoll, process_data, process_names, sig_process_data, classifier_sigeff_range = (1.0, 0.0), nJ = 2):
        
        # first, determine the cuts on the classifier based on the asked-for signal efficiency
        classifier_range = ClassifierBasedCategoryFiller._sigeff_range_to_score_range(mcoll, sig_process_data, sigeff_range = classifier_sigeff_range)
        print("translated signal efficiency range ({}, {}) to classifier output range ({}, {})".format(classifier_sigeff_range[0], classifier_sigeff_range[1], 
                                                                                                       classifier_range[0], classifier_range[1]))
        
        retcat = Category("clf_{:.2f}_{:.2f}".format(classifier_sigeff_range[0], classifier_sigeff_range[1]))

        for cur_process_data, cur_process_name in zip(process_data, process_names):

            cur_process_data = cur_process_data.loc[cur_process_data["nJ"] == nJ]
            
            print("predicting on sample {} with length {}".format(cur_process_name, len(cur_process_data)))

            cur_pred = mcoll.predict(cur_process_data)[:, 1]
            cut = np.logical_and.reduce((cur_pred > classifier_range[0], cur_pred < classifier_range[1]))

            assert len(cut) == len(cur_process_data)
            passed = cur_process_data[cut]
            passed = TrainingSample.fromTable(passed)
            
            # fill the category
            retcat.add_events(events = passed.data, weights = passed.weights, process = cur_process_name, event_variables = TrainingConfig.training_branches)

        return retcat
