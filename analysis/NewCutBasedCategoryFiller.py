import numpy as np
from analysis.Category import Category
from base.Configs import TrainingConfig
from training.DataFormatters import TrainingSample, only_nJ

class CutBasedCategoryFiller:

    @staticmethod
    def create_nJ_category(process_data, process_names, nJ = 2):
        
        retcat = Category("inclusive_{}J".format(nJ))
        formatter = only_nJ(nJ = nJ)

        for cur_process_data, cur_process_name in zip(process_data, process_names):
            
            passed = formatter.format_as_TrainingSample(cur_process_data)
            retcat.add_events(events = passed.data, weights = passed.weights, process = cur_process_name, event_variables = TrainingConfig.training_branches)

        return retcat

    @staticmethod
    def create_low_MET_category(process_data, process_names, nJ = 2, cuts = {"MET_cut": 191, "dRBB_highMET_cut": 1.2, "dRBB_lowMET_cut": 5.0}):
        
        retcat = Category("low_MET")

        for cur_process_data, cur_process_name in zip(process_data, process_names):
            
            # apply the cuts
            passed = cur_process_data.loc[(cur_process_data["MET"] > 150) & (cur_process_data["MET"] < cuts["MET_cut"]) & (cur_process_data["dRBB"] < cuts["dRBB_lowMET_cut"]) & (cur_process_data["nJ"] == nJ)]
            passed = TrainingSample.fromTable(passed)

            # fill the category
            retcat.add_events(events = passed.data, weights = passed.weights, process = cur_process_name, event_variables = TrainingConfig.training_branches)

            print("filled {} events from process '{}'".format(sum(passed.weights), cur_process_name))

        return retcat

    @staticmethod
    def create_high_MET_category(process_data, process_names, nJ = 2, cuts = {"MET_cut": 191, "dRBB_highMET_cut": 1.2, "dRBB_lowMET_cut": 5.0}):

        retcat = Category("high_MET")

        for cur_process_data, cur_process_name in zip(process_data, process_names):
            
            # apply the cuts
            passed = cur_process_data.loc[(cur_process_data["MET"] > cuts["MET_cut"]) & (cur_process_data["dRBB"] < cuts["dRBB_highMET_cut"]) & (cur_process_data["nJ"] == nJ)]
            passed = TrainingSample.fromTable(passed)

            # fill the category
            retcat.add_events(events = passed.data, weights = passed.weights, process = cur_process_name, event_variables = TrainingConfig.training_branches)

            print("filled {} events from process '{}'".format(sum(passed.weights), cur_process_name))

        return retcat

