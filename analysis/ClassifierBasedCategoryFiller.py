import numpy as np

from analysis.Category import Category
from plotting.ModelEvaluator import ModelEvaluator
from base.Configs import TrainingConfig

class ClassifierBasedCategoryFiller:

    @staticmethod
    def _sigeff_to_score(env, signal_events, signal_weights, sigeff):
        signal_events = np.concatenate(signal_events)
        signal_weights = np.concatenate(signal_weights)
        signal_pred = env.predict(data = signal_events)[:,1] # obtain the prediction of the model
        return ModelEvaluator._weighted_percentile(signal_pred, 1 - sigeff, weights = signal_weights)        

    @staticmethod
    def _score_to_sigeff(env, signal_events, signal_weights, score):
        signal_events = np.concatenate(signal_events)
        signal_weights = np.concatenate(signal_weights)
        signal_pred = env.predict(data = signal_events)[:,1] # obtain the prediction of the model
        
        total_passed = np.sum[signal_weights[signal_pred > score]]
        total_events = np.sum[signal_weights]

        return total_passed / total_events

    @staticmethod
    def create_classifier_category(env, process_events, process_aux_events, process_weights, process_names, signal_events, signal_weights, classifier_sigeff_range = (1, 0), nJ = 2, interpret_as_sigeff = True, process_preds = None):

        if not process_preds:
            process_preds = [None for cur_events in process_events]

        if interpret_as_sigeff:
            if(classifier_sigeff_range[0] < classifier_sigeff_range[1]):
                raise Exception("Warning: are you sure you understand what these cuts are doing? Lower signal efficiencies correspond to _harsher_ cuts, so expect (higher number, lower number)!")
            
            # first, compute the cut values that correspond to the given signal efficiency values
            classifier_range = (ClassifierBasedCategoryFiller._sigeff_to_score(env, signal_events, signal_weights, classifier_sigeff_range[0]),
                                ClassifierBasedCategoryFiller._sigeff_to_score(env, signal_events, signal_weights, classifier_sigeff_range[1]))

            print("translated signal efficiency range ({}, {}) to classifier output range ({}, {})".format(classifier_sigeff_range[0], classifier_sigeff_range[1], 
                                                                                                           classifier_range[0], classifier_range[1]))
        else:
            classifier_range = classifier_sigeff_range

        retcat = Category("clf_{:.2f}_{:.2f}".format(classifier_sigeff_range[0], classifier_sigeff_range[1]))
        
        for cur_events, cur_aux_events, cur_weights, process_name, cur_pred in zip(process_events, process_aux_events, process_weights, process_names, process_preds):
            # get the classifier predictions
            if cur_pred is None:
                cur_pred = env.predict(data = cur_events)[:,1]

            cur_nJ = cur_aux_events[:, TrainingConfig.other_branches.index("nJ")]

            if nJ:
                # a cut on the number of jets was requested
                cut = np.logical_and.reduce((cur_pred > classifier_range[0], cur_pred < classifier_range[1], cur_nJ == nJ))
            else:
                # fill this category inclusively in the number of jets
                cut = np.logical_and.reduce((cur_pred > classifier_range[0], cur_pred < classifier_range[1]))

            passed_events = cur_events[cut]
            passed_weights = cur_weights[cut]

            # also store some auxiliary information in this category
            aux_content = np.expand_dims(cur_pred[cut], axis = 1)
            aux_variables = ["clf"]

            retcat.add_events(events = passed_events, weights = passed_weights, process = process_name, event_variables = TrainingConfig.training_branches,
                              aux_content = aux_content, aux_variables = aux_variables)

        return retcat
