from configparser import ConfigParser
import os

class TrainingConfig:
    # branches to use for the training
    training_branches = ["mBB", "dRBB", "pTB1", "pTB2", "MET", "dEtaBB", "dPhiMETdijet", "SumPtJet"]
    #training_branches = ["pTB1", "pTB2", "MET", "dEtaBB", "dPhiMETdijet", "SumPtJet"]
    nuisance_branches = ["mBB"]
    auxiliary_branches = ["EventWeight", "mBB", "dRBB", "nJ"]
    #other_branches = ["nJ"]  # any other branches that might be necessary for purposes other than training and evaluating a classifier

    training_pars = {"pretrain_batches": 100, "training_batches": 800, "printout_interval": 10}

    training_slice = [0.0, 0.33]
    test_slice = [0.33, 0.66]
    validation_slice = [0.66, 1.0]

    #data_path = "/home/windischhofer/datasmall/Hbb/training-MadGraphPy8-ATLAS-filtered.h5"
    data_path = "/home/windischhofer/datasmall/Hbb/training-MadGraphPy8-ATLAS.h5"
    #data_path = "/home/windischhofer/datasmall/Hbb/training-mc16d.h5"

    sig_samples = ["Hbb"]
    sig_sampling_lengths = [1.0]

    bkg_samples = ["Zjets", "Wjets", "ttbar", "diboson"]
    bkg_sampling_lengths = [1.0, 1.0, 1.0, 1.0]

    # bkg_samples = ["ttbar"]
    # bkg_sampling_lengths = [1.0]

    sample_reweighting = {"Hbb": 1.0, "Zjets": 1.0, "Wjets": 1.0, "ttbar": 1.0, "diboson": 1.0}

    @classmethod
    def from_file(cls, config_dir):
        gconfig = ConfigParser()
        gconfig.read(os.path.join(config_dir, "meta.conf"))

        cur_pars = {key: float(val) for key, val in gconfig["TrainingConfig"].items()}

        obj = cls()
        obj.training_pars.update(cur_pars)

        return obj        
