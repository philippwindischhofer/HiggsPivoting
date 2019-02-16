import numpy as np

from analysis.Category import Category
from plotting.ModelEvaluator import ModelEvaluator

class ClassifierBasedCategoryFiller:

    @staticmethod
    def create_classifier_category(env, process_events, process_weights, process_names, signal_events, signal_weights, classifier_sigeff_range = (1, 0)):
        if(classifier_sigeff_range[0] < classifier_sigeff_range[1]):
            raise Exception("Warning: are you sure you understand what these cuts are doing? Lower signal efficiencies correspond to _harsher_ cuts, so expect (higher number, lower number)!")

        retcat = Category("clf_{:.2f}_{:.2f}".format(classifier_sigeff_range[0], classifier_sigeff_range[1]))

        # first, compute the cut values that correspond to the given signal efficiency values
        signal_events = np.concatenate(signal_events)
        signal_weights = np.concatenate(signal_weights)
        signal_pred = env.predict(data = signal_events)[:,1] # obtain the prediction of the model
        classifier_range = (ModelEvaluator._weighted_percentile(signal_pred, 1 - classifier_sigeff_range[0], weights = signal_weights),
                            ModelEvaluator._weighted_percentile(signal_pred, 1 - classifier_sigeff_range[1], weights = signal_weights))

        print("translated signal efficiency range ({}, {}) to classifier output range ({}, {})".format(classifier_sigeff_range[0], classifier_sigeff_range[1], 
                                                                                                       classifier_range[0], classifier_range[1]))
        
        for cur_events, cur_weights, process_name in zip(process_events, process_weights, process_names):
            # get the classifier predictions
            cur_pred = env.predict(data = cur_events)[:,1]

            cut = np.logical_and.reduce((cur_pred > classifier_range[0], cur_pred < classifier_range[1]))

            passed_events = cur_events[cut]
            passed_weights = cur_weights[cut]

            retcat.add_events(events = passed_events, weights = passed_weights, process = process_name)

        return retcat
