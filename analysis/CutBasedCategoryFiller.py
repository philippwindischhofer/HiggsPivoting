import numpy as np

from base.Configs import TrainingConfig
from analysis.Category import Category

# fills the event categories used in the cut-based cross-check analysis for VHbb
class CutBasedCategoryFiller:
    
    # cuts applied to incoming events:
    # MET > 150 GeV && MET < 200 GeV
    # dRBB < 1.8
    @staticmethod
    def create_low_MET_category(process_events, process_aux_events, process_weights, process_names, nJ = 2):
        retcat = Category("low_MET")

        for cur_events, cur_aux_events, cur_weights, process_name in zip(process_events, process_aux_events, process_weights, process_names):
            # extract the branches that are needed for the cut
            cur_MET = cur_events[:, TrainingConfig.training_branches.index("MET")]
            cur_dRBB = cur_events[:, TrainingConfig.training_branches.index("dRBB")]

            cur_nJ = cur_aux_events[:, TrainingConfig.other_branches.index("nJ")]

            cut = np.logical_and.reduce((cur_MET > 150, cur_MET < 200, cur_dRBB < 1.8, cur_nJ == nJ))

            passed_events = cur_events[cut]
            passed_weights = cur_weights[cut]

            retcat.add_events(events = passed_events, weights = passed_weights, process = process_name, event_variables = TrainingConfig.training_branches)

        return retcat

    # cuts applied to incoming events:
    # MET > 200 GeV
    # dRBB < 1.2
    @staticmethod
    def create_high_MET_category(process_events, process_aux_events, process_weights, process_names, nJ = 2):
        retcat = Category("high_MET")

        for cur_events, cur_aux_events, cur_weights, process_name in zip(process_events, process_aux_events, process_weights, process_names):
            # extract the branches that are needed for the cut
            cur_MET = cur_events[:, TrainingConfig.training_branches.index("MET")]
            cur_dRBB = cur_events[:, TrainingConfig.training_branches.index("dRBB")]

            cur_nJ = cur_aux_events[:, TrainingConfig.other_branches.index("nJ")]

            cut = np.logical_and.reduce((cur_MET > 200, cur_dRBB < 1.2, cur_nJ == nJ))

            passed_events = cur_events[cut]
            passed_weights = cur_weights[cut]

            retcat.add_events(events = passed_events, weights = passed_weights, process = process_name, event_variables = TrainingConfig.training_branches)

        return retcat
